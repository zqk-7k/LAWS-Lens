#!/usr/bin/env python3
"""Initialize the sky-background v10 experiment and stop before authorization.

This script implements only protocol stage G-1. It never launches data inventory,
power analysis, injections, PE, calibration, scoring, or real-catalog reranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
CONFIG_KINDS = (
    "DESIGN_CONFIG_DRAFT",
    "PE_CONFIG",
    "INJECTION_AND_SELECTION_CONFIG",
    "V93_BASELINE_CONTRACT",
    "ANALYSIS_CONFIG_FINAL",
)
INITIAL_AUTHOR_CONFIGS = CONFIG_KINDS[:4]
HOLD_STATUS = "HOLD_FOR_AUTHOR_SIGNATURE"
JCS_EXPECTED_CANONICAL = (
    '{"ascii":"configuration","float_scientific":1e-7,'
    '"float_small":0.000001,"nested":{"a":{"count":10000,'
    '"threshold":0.05},"z":[true,null,0]},"unicode":"天空后验"}'
)
JCS_EXPECTED_SHA256 = "ead281ad1bafe9430cab45ec834338d58b867cb471ab42f9402390815b080900"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _validate_unicode(value: str) -> None:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError("JCS input contains an unpaired Unicode surrogate")


def _jcs_number(value: int | float) -> str:
    if isinstance(value, bool):
        raise TypeError("bool is not a JCS number")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise ValueError("JCS forbids NaN and Infinity")
    if value == 0:
        return "0"

    magnitude = abs(value)
    shortest = repr(value).lower()
    if 1e-6 <= magnitude < 1e21:
        if "e" in shortest:
            mantissa, exponent = shortest.split("e")
            sign = "-" if mantissa.startswith("-") else ""
            digits = mantissa.lstrip("-").replace(".", "")
            decimal_places = len(mantissa.lstrip("-").split(".")[1]) if "." in mantissa else 0
            shift = int(exponent) - decimal_places
            if shift >= 0:
                result = sign + digits + "0" * shift
            else:
                split_at = len(digits) + shift
                if split_at > 0:
                    result = sign + digits[:split_at] + "." + digits[split_at:]
                else:
                    result = sign + "0." + "0" * (-split_at) + digits
        else:
            result = shortest
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        return result

    if "e" not in shortest:
        shortest = format(value, ".15e")
    mantissa, exponent = shortest.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp_int = int(exponent)
    return f"{mantissa}e{'+' if exp_int >= 0 else ''}{exp_int}"


def jcs_bytes(value: Any) -> bytes:
    """Canonicalize the JSON subset used by experiment contracts.

    The implementation follows RFC 8785 ordering, string escaping, finite-number,
    and whitespace rules. A checked test vector is emitted with every run.
    """

    def encode(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, (int, float)):
            return _jcs_number(item)
        if isinstance(item, str):
            _validate_unicode(item)
            return json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if isinstance(item, list):
            return "[" + ",".join(encode(element) for element in item) + "]"
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("JCS object keys must be strings")
            for key in item:
                _validate_unicode(key)
            keys = sorted(item, key=lambda key: key.encode("utf-16-be"))
            return "{" + ",".join(f"{encode(key)}:{encode(item[key])}" for key in keys) + "}"
        raise TypeError(f"Unsupported JCS value: {type(item).__name__}")

    return encode(value).encode("utf-8")


def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(jcs_bytes(payload)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def base_schema(kind: str, required_sections: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://gw-catalog.local/schemas/{kind.lower()}.schema.json",
        "title": f"{kind} contract",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "config_kind",
            "lifecycle_status",
            "approved_for",
            "created_at_utc",
            *required_sections,
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "config_kind": {"const": kind},
            "lifecycle_status": {"enum": ["UNSIGNED_TEMPLATE", "AUTHOR_APPROVED"]},
            "approved_for": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "created_at_utc": {"type": "string", "format": "date-time"},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
    }


def build_schemas() -> dict[str, dict[str, Any]]:
    design = base_schema("DESIGN_CONFIG_DRAFT", ["study", "candidate_arms", "gate_proposals", "power_targets", "resource_limits"])
    design["properties"].update({
        "study": {"type": "object", "required": ["objective", "deployments", "status"], "properties": {
            "objective": {"type": "string"},
            "deployments": {"type": "array", "items": {"enum": ["GWTC3_O3", "GWTC4P1_O4A"]}, "minItems": 2, "uniqueItems": True},
            "status": {"const": "EXPLORATORY_NO_AUTOMATIC_ADOPTION"},
        }, "additionalProperties": False},
        "candidate_arms": {"type": "array", "items": {"type": "string"}, "minItems": 7, "uniqueItems": True},
        "gate_proposals": {"type": "object", "minProperties": 6},
        "power_targets": {"type": "object", "required": ["monte_carlo_replicates", "minimum_joint_pass_probability", "maximum_false_upgrade_probability"], "properties": {
            "monte_carlo_replicates": nullable({"type": "integer", "minimum": 10000}),
            "minimum_joint_pass_probability": nullable({"type": "number", "minimum": 0, "maximum": 1}),
            "maximum_false_upgrade_probability": nullable({"type": "number", "minimum": 0, "maximum": 1}),
            "planning_data_paths": {"type": "array", "items": {"type": "string"}},
        }, "additionalProperties": False},
        "resource_limits": {"type": "object", "required": ["max_pe_events", "max_cpu_hours", "max_gpu_hours", "max_storage_gb", "max_wallclock_days"], "properties": {
            "max_pe_events": nullable({"type": "integer", "minimum": 1}),
            "max_cpu_hours": nullable({"type": "number", "exclusiveMinimum": 0}),
            "max_gpu_hours": nullable({"type": "number", "minimum": 0}),
            "max_storage_gb": nullable({"type": "number", "exclusiveMinimum": 0}),
            "max_wallclock_days": nullable({"type": "number", "exclusiveMinimum": 0}),
        }, "additionalProperties": False},
        "notes": {"type": "array", "items": {"type": "string"}},
    })

    pe = base_schema("PE_CONFIG", ["pipeline", "data", "psd", "sampler", "calibration", "skymap"])
    pe["properties"].update({
        "pipeline": {"type": "object", "required": ["pipeline_id", "software_container_digest", "waveform_approximant", "prior_path", "prior_sha256"], "properties": {
            "pipeline_id": nullable({"type": "string", "minLength": 1}),
            "software_container_digest": nullable({"type": "string", "minLength": 16}),
            "waveform_approximant": nullable({"type": "string", "minLength": 1}),
            "prior_path": nullable({"type": "string", "minLength": 1}),
            "prior_sha256": nullable({"type": "string", "pattern": "^[0-9a-f]{64}$"}),
        }, "additionalProperties": False},
        "data": {"type": "object", "required": ["sampling_frequency_hz", "segment_length_seconds", "fmin_hz", "fmax_hz", "detector_policy", "data_quality_rule", "guard_window_seconds"], "properties": {
            "sampling_frequency_hz": nullable({"type": "number", "exclusiveMinimum": 0}),
            "segment_length_seconds": nullable({"type": "number", "exclusiveMinimum": 0}),
            "fmin_hz": nullable({"type": "number", "minimum": 0}),
            "fmax_hz": nullable({"type": "number", "exclusiveMinimum": 0}),
            "detector_policy": nullable({"type": "string", "minLength": 1}),
            "data_quality_rule": nullable({"type": "string", "minLength": 1}),
            "guard_window_seconds": nullable({"type": "number", "minimum": 0}),
        }, "additionalProperties": False},
        "psd": {"type": "object", "required": ["method", "offsource_duration_seconds", "window", "overlap_fraction"], "properties": {
            "method": nullable({"type": "string", "minLength": 1}),
            "offsource_duration_seconds": nullable({"type": "number", "exclusiveMinimum": 0}),
            "window": nullable({"type": "string", "minLength": 1}),
            "overlap_fraction": nullable({"type": "number", "minimum": 0, "maximum": 1}),
        }, "additionalProperties": False},
        "sampler": {"type": "object", "required": ["name", "settings", "convergence_rule", "random_seed_policy"], "properties": {
            "name": nullable({"type": "string", "minLength": 1}),
            "settings": nullable({"type": "object"}),
            "convergence_rule": nullable({"type": "object"}),
            "random_seed_policy": nullable({"type": "string", "minLength": 1}),
        }, "additionalProperties": False},
        "calibration": {"type": "object", "required": ["marginalization_enabled", "model"], "properties": {
            "marginalization_enabled": nullable({"type": "boolean"}),
            "model": nullable({"type": "string"}),
        }, "additionalProperties": False},
        "skymap": {"type": "object", "required": ["analysis_nside", "convergence_nsides", "ordering", "coordinate_frame", "value_convention"], "properties": {
            "analysis_nside": {"const": 1024},
            "convergence_nsides": {"const": [256, 512, 1024]},
            "ordering": nullable({"enum": ["RING", "NESTED"]}),
            "coordinate_frame": nullable({"type": "string", "minLength": 1}),
            "value_convention": nullable({"enum": ["probability_mass", "probability_density"]}),
        }, "additionalProperties": False},
        "notes": {"type": "array", "items": {"type": "string"}},
    })

    injection = base_schema("INJECTION_AND_SELECTION_CONFIG", ["populations", "waveform", "lensing", "detector_response", "selection", "random_seeds"])
    injection["properties"].update({
        "populations": {"type": "object", "required": ["source_population", "lens_population", "unlensed_population", "parameter_distributions"], "properties": {
            "source_population": nullable({"type": "string", "minLength": 1}),
            "lens_population": nullable({"type": "string", "minLength": 1}),
            "unlensed_population": nullable({"type": "string", "minLength": 1}),
            "parameter_distributions": nullable({"type": "object"}),
        }, "additionalProperties": False},
        "waveform": {"type": "object", "required": ["approximant", "generation_code_path", "generation_code_sha256"], "properties": {
            "approximant": nullable({"type": "string", "minLength": 1}),
            "generation_code_path": nullable({"type": "string", "minLength": 1}),
            "generation_code_sha256": nullable({"type": "string", "pattern": "^[0-9a-f]{64}$"}),
        }, "additionalProperties": False},
        "lensing": {"type": "object", "required": ["delay_distribution", "magnification_distribution", "image_type_distribution", "double_quad_ratio", "morse_phase_rule"], "properties": {
            "delay_distribution": nullable({"type": "object"}),
            "magnification_distribution": nullable({"type": "object"}),
            "image_type_distribution": nullable({"type": "object"}),
            "double_quad_ratio": nullable({"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 2, "maxItems": 2}),
            "morse_phase_rule": nullable({"type": "string", "minLength": 1}),
        }, "additionalProperties": False},
        "detector_response": {"type": "object", "required": ["runs", "antenna_response", "local_psd", "common_system_amplitude_scaling"], "properties": {
            "runs": {"type": "array", "items": {"type": "string"}},
            "antenna_response": nullable({"type": "string", "minLength": 1}),
            "local_psd": nullable({"type": "string", "minLength": 1}),
            "common_system_amplitude_scaling": nullable({"type": "boolean"}),
        }, "additionalProperties": False},
        "selection": {"type": "object", "required": ["search_pipeline", "network_snr_threshold", "single_detector_thresholds", "live_interval_rule", "data_quality_rule", "both_images_detected_rule"], "properties": {
            "search_pipeline": nullable({"type": "string", "minLength": 1}),
            "network_snr_threshold": nullable({"type": "number", "minimum": 0}),
            "single_detector_thresholds": nullable({"type": "object"}),
            "live_interval_rule": nullable({"type": "string", "minLength": 1}),
            "data_quality_rule": nullable({"type": "string", "minLength": 1}),
            "both_images_detected_rule": nullable({"type": "boolean"}),
        }, "additionalProperties": False},
        "random_seeds": {"type": "object", "required": ["source", "lens", "noise", "injection", "pe", "catalog_assignment"], "properties": {
            key: {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}
            for key in ("source", "lens", "noise", "injection", "pe", "catalog_assignment")
        }, "additionalProperties": False},
        "notes": {"type": "array", "items": {"type": "string"}},
    })

    baseline = base_schema("V93_BASELINE_CONTRACT", ["historical_root", "deployments", "frozen_scoring", "strict_scope"])
    baseline["properties"].update({
        "historical_root": {"type": "string", "minLength": 1},
        "deployments": {"type": "object", "required": ["GWTC3_O3", "GWTC4P1_O4A"], "properties": {
            deployment: {"type": "object", "required": ["model_seeds", "model_paths", "waveform_score_inputs", "time_score_inputs", "weights_per_seed", "strict_scope_manifest"], "properties": {
                "model_seeds": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3, "uniqueItems": True},
                "model_paths": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
                "waveform_score_inputs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "time_score_inputs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "weights_per_seed": {"type": "array", "items": {"type": "object"}, "minItems": 3, "maxItems": 3},
                "strict_scope_manifest": nullable({"type": "string", "minLength": 1}),
            }, "additionalProperties": False}
            for deployment in ("GWTC3_O3", "GWTC4P1_O4A")
        }, "additionalProperties": False},
        "frozen_scoring": {"type": "object", "required": ["waveform_definition", "time_definition", "sky_definition", "consensus_definition", "ties_rule"], "properties": {
            "waveform_definition": nullable({"type": "string", "minLength": 1}),
            "time_definition": nullable({"type": "string", "minLength": 1}),
            "sky_definition": {"const": "Z_sky=log(Npix*sum(P_i*P_j))"},
            "consensus_definition": nullable({"type": "string", "minLength": 1}),
            "ties_rule": nullable({"type": "string", "minLength": 1}),
        }, "additionalProperties": False},
        "strict_scope": {"type": "object", "required": ["gwtc3_event_count", "o4a_event_count", "h1_l1_complete_required", "bbh_only"], "properties": {
            "gwtc3_event_count": {"const": 53},
            "o4a_event_count": {"const": 74},
            "h1_l1_complete_required": {"const": True},
            "bbh_only": {"const": True},
        }, "additionalProperties": False},
        "input_hash_manifest": nullable({"type": "string", "minLength": 1}),
        "notes": {"type": "array", "items": {"type": "string"}},
    })

    final = base_schema("ANALYSIS_CONFIG_FINAL", ["recommendation_hash", "sample_sizes", "calibration", "exploratory_arms", "gates", "resource_budget"])
    final["properties"].update({
        "recommendation_hash": nullable({"type": "string", "pattern": "^[0-9a-f]{64}$"}),
        "sample_sizes": nullable({"type": "object"}),
        "calibration": nullable({"type": "object"}),
        "exploratory_arms": {"const": [
            "v93_same_domain", "one_stage_eta_000", "one_stage_eta_025",
            "one_stage_eta_050", "one_stage_eta_100", "two_stage_wt_only", "two_stage_full",
        ]},
        "gates": nullable({"type": "object"}),
        "resource_budget": nullable({"type": "object"}),
        "notes": {"type": "array", "items": {"type": "string"}},
    })

    approval = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://gw-catalog.local/schemas/approval_sidecar.schema.json",
        "title": "Detached author approval sidecar",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "payload_filename", "payload_sha256_jcs", "approval_status", "approved_for", "approved_at_utc", "author_identity", "signature_method", "signature"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "payload_filename": {"type": "string", "minLength": 1},
            "payload_sha256_jcs": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "approval_status": {"enum": ["UNSIGNED", "AUTHOR_APPROVED"]},
            "approved_for": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "approved_at_utc": nullable({"type": "string", "format": "date-time"}),
            "author_identity": nullable({"type": "string", "minLength": 1}),
            "signature_method": {"enum": ["UNSIGNED", "MANUAL_DETACHED_ATTESTATION", "GPG_DETACHED", "MINISIGN_ED25519"]},
            "signature": nullable({"type": "string", "minLength": 1}),
            "public_key_fingerprint": nullable({"type": "string", "minLength": 1}),
        },
    }
    return {
        "DESIGN_CONFIG_DRAFT.schema.json": design,
        "PE_CONFIG.schema.json": pe,
        "INJECTION_AND_SELECTION_CONFIG.schema.json": injection,
        "V93_BASELINE_CONTRACT.schema.json": baseline,
        "ANALYSIS_CONFIG_FINAL.schema.json": final,
        "APPROVAL_SIDECAR.schema.json": approval,
    }


def template_base(kind: str, created: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_kind": kind,
        "lifecycle_status": "UNSIGNED_TEMPLATE",
        "approved_for": [],
        "created_at_utc": created,
    }


def build_templates(project_root: Path, created: str) -> dict[str, dict[str, Any]]:
    design = template_base("DESIGN_CONFIG_DRAFT", created)
    design.update({
        "study": {
            "objective": "Explore run-matched conditional-sky calibration and non-compensatory reranking without replacing v9.3.",
            "deployments": ["GWTC3_O3", "GWTC4P1_O4A"],
            "status": "EXPLORATORY_NO_AUTOMATIC_ADOPTION",
        },
        "candidate_arms": [
            "v93_same_domain", "one_stage_eta_000", "one_stage_eta_025",
            "one_stage_eta_050", "one_stage_eta_100", "two_stage_wt_only", "two_stage_full",
        ],
        "gate_proposals": {
            "A_R10_noninferiority_lcb": -0.005,
            "A_per_seed_point_floor": -0.015,
            "B_F50_point_reduction": 0.10,
            "B_F50_lcb_reduction": 0.05,
            "C_F90_ratio_ucb": 1.05,
            "D_AUPRC_ratio_lcb": 0.98,
            "E_nominal_1pct_fpr_ucb": 0.015,
            "E_nominal_5pct_fpr_ucb": 0.06,
            "F_sky_F50_point_reduction": 0.05,
        },
        "power_targets": {
            "monte_carlo_replicates": None,
            "minimum_joint_pass_probability": None,
            "maximum_false_upgrade_probability": None,
            "planning_data_paths": [],
        },
        "resource_limits": {
            "max_pe_events": None,
            "max_cpu_hours": None,
            "max_gpu_hours": None,
            "max_storage_gb": None,
            "max_wallclock_days": None,
        },
        "notes": ["AUTHOR MUST FILL POWER ASSUMPTIONS AND RESOURCE LIMITS BEFORE APPROVAL."],
    })

    pe = template_base("PE_CONFIG", created)
    pe.update({
        "pipeline": {"pipeline_id": None, "software_container_digest": None, "waveform_approximant": None, "prior_path": None, "prior_sha256": None},
        "data": {"sampling_frequency_hz": None, "segment_length_seconds": None, "fmin_hz": None, "fmax_hz": None, "detector_policy": None, "data_quality_rule": None, "guard_window_seconds": None},
        "psd": {"method": None, "offsource_duration_seconds": None, "window": None, "overlap_fraction": None},
        "sampler": {"name": None, "settings": None, "convergence_rule": None, "random_seed_policy": None},
        "calibration": {"marginalization_enabled": None, "model": None},
        "skymap": {"analysis_nside": 1024, "convergence_nsides": [256, 512, 1024], "ordering": None, "coordinate_frame": None, "value_convention": None},
        "notes": ["NO HOMOGENEOUS PE MAY START UNTIL THIS PAYLOAD IS AUTHOR APPROVED."],
    })

    injection = template_base("INJECTION_AND_SELECTION_CONFIG", created)
    injection.update({
        "populations": {"source_population": None, "lens_population": None, "unlensed_population": None, "parameter_distributions": None},
        "waveform": {"approximant": None, "generation_code_path": None, "generation_code_sha256": None},
        "lensing": {"delay_distribution": None, "magnification_distribution": None, "image_type_distribution": None, "double_quad_ratio": None, "morse_phase_rule": None},
        "detector_response": {"runs": ["O1", "O2", "O3a", "O3b", "O4a"], "antenna_response": None, "local_psd": None, "common_system_amplitude_scaling": None},
        "selection": {"search_pipeline": None, "network_snr_threshold": None, "single_detector_thresholds": None, "live_interval_rule": None, "data_quality_rule": None, "both_images_detected_rule": None},
        "random_seeds": {key: [] for key in ("source", "lens", "noise", "injection", "pe", "catalog_assignment")},
        "notes": ["SOURCE/LENS IDS MUST BE GLOBALLY DISJOINT ACROSS ALL HISTORICAL AND NEW SPLITS."],
    })

    baseline = template_base("V93_BASELINE_CONTRACT", created)
    baseline.update({
        "historical_root": str(project_root / "results/gwtc_sky_resolution_v93_20260730"),
        "deployments": {
            deployment: {
                "model_seeds": [], "model_paths": [], "waveform_score_inputs": [],
                "time_score_inputs": [], "weights_per_seed": [], "strict_scope_manifest": None,
            }
            for deployment in ("GWTC3_O3", "GWTC4P1_O4A")
        },
        "frozen_scoring": {
            "waveform_definition": None,
            "time_definition": None,
            "sky_definition": "Z_sky=log(Npix*sum(P_i*P_j))",
            "consensus_definition": None,
            "ties_rule": None,
        },
        "strict_scope": {"gwtc3_event_count": 53, "o4a_event_count": 74, "h1_l1_complete_required": True, "bbh_only": True},
        "input_hash_manifest": None,
        "notes": ["CANDIDATE PATHS MUST BE RESOLVED AND HASHED, THEN AUTHOR APPROVED."],
    })

    final = template_base("ANALYSIS_CONFIG_FINAL", created)
    final.update({
        "recommendation_hash": None,
        "sample_sizes": None,
        "calibration": None,
        "exploratory_arms": [
            "v93_same_domain", "one_stage_eta_000", "one_stage_eta_025",
            "one_stage_eta_050", "one_stage_eta_100", "two_stage_wt_only", "two_stage_full",
        ],
        "gates": None,
        "resource_budget": None,
        "notes": ["DO NOT SIGN BEFORE G0.5 RECOMMENDATION AND AUTHOR REVIEW."],
    })
    return {f"{kind}.json": value for kind, value in zip(CONFIG_KINDS, (design, pe, injection, baseline, final))}


def approval_template(filename: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "payload_filename": filename,
        "payload_sha256_jcs": payload_hash(payload),
        "approval_status": "UNSIGNED",
        "approved_for": [],
        "approved_at_utc": None,
        "author_identity": None,
        "signature_method": "UNSIGNED",
        "signature": None,
        "public_key_fingerprint": None,
    }


def build_readme(run_id: str) -> str:
    return f"""# G-1 初始化结果索引

Run ID: `{run_id}`

本目录严格停在 G-1。它只包含 JSON Schema、未签名配置模板、签名校验工具和审计记录。

## 当前状态

`{HOLD_STATUS}`

未执行：source/noise inventory、G0.5 power audit、注入、PE、天空校准、holdout、真实候选重排。

## 关键文件

- `contracts/protocol/真实运行期匹配天空背景与非补偿重排实施协议_20260822_CN.md`
- `contracts/schemas/`: 六份 JSON Schema
- `contracts/unsigned_templates/`: 五份配置 payload 及 detached approval sidecar
- `contracts/canonicalization/`: JCS 测试向量、canonical bytes 和 SHA-256
- `provenance/G_MINUS1_CONFIG_AUDIT.json`
- `reports/G_MINUS1_REPORT_CN.md`
- `reports/AUTHOR_ACTION_REQUIRED_CN.md`
- `scripts/00a_generate_json_schemas_and_unsigned_templates_then_hold.py`
- `scripts/00_validate_author_signed_design_configs.py`

下一步不是启动计算，而是作者填写并签署前四份配置。`ANALYSIS_CONFIG_FINAL` 只能在 G0.5 后签署。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    protocol = args.protocol.resolve()
    output = args.output_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")

    created = utc_now()
    run_id = output.name
    directories = [
        "contracts/protocol", "contracts/schemas", "contracts/unsigned_templates",
        "contracts/canonicalization", "provenance", "calibration/one_stage",
        "calibration/two_stage", "injections/o3", "injections/o4a",
        "scores/baseline_v93_same_domain", "scores/one_stage_eta_000",
        "scores/one_stage_eta_025", "scores/one_stage_eta_050",
        "scores/one_stage_eta_100", "scores/two_stage_wt_only",
        "scores/two_stage_full", "real_catalog/descriptive_only", "tables",
        "figures", "reports", "logs", "scripts", "checkpoints", "package",
    ]
    for directory in directories:
        (output / directory).mkdir(parents=True, exist_ok=False)

    shutil.copy2(protocol, output / "contracts/protocol" / protocol.name)
    shutil.copy2(Path(__file__).resolve(), output / "scripts" / Path(__file__).name)
    validator = Path(__file__).resolve().with_name("00_validate_author_signed_design_configs.py")
    if not validator.is_file():
        raise FileNotFoundError(f"Missing sibling validator: {validator}")
    shutil.copy2(validator, output / "scripts" / validator.name)

    schemas = build_schemas()
    for filename, schema in schemas.items():
        write_json(output / "contracts/schemas" / filename, schema)

    templates = build_templates(project_root, created)
    for filename, payload in templates.items():
        target = output / "contracts/unsigned_templates" / filename
        write_json(target, payload)
        write_json(target.with_suffix(".approval.json"), approval_template(filename, payload))

    vector = {
        "ascii": "configuration",
        "nested": {"z": [True, None, -0.0], "a": {"threshold": 0.05, "count": 10000}},
        "unicode": "天空后验",
        "float_small": 0.000001,
        "float_scientific": 1e-7,
    }
    canonical = jcs_bytes(vector)
    canonical_hash = hashlib.sha256(canonical).hexdigest()
    if canonical != JCS_EXPECTED_CANONICAL.encode("utf-8") or canonical_hash != JCS_EXPECTED_SHA256:
        raise RuntimeError("RFC 8785 canonicalization test vector failed")
    write_json(output / "contracts/canonicalization/JCS_TEST_VECTOR.json", vector)
    (output / "contracts/canonicalization/JCS_TEST_VECTOR.canonical.json").write_bytes(canonical + b"\n")
    (output / "contracts/canonicalization/JCS_TEST_VECTOR.sha256").write_text(canonical_hash + "\n", encoding="ascii")

    git_commit = command_output(["git", "rev-parse", "HEAD"], project_root)
    git_status = command_output(["git", "status", "--porcelain"], project_root)
    v93_root = project_root / "results/gwtc_sky_resolution_v93_20260730"
    v93_contract = v93_root / "analysis_contract_v93.json"
    v93_manifest = v93_root / "checksums_sha256.txt"
    initial_audit = {
        "schema": "sky-background-v10-g-minus1-audit-v1",
        "run_id": run_id,
        "created_at_utc": created,
        "status": HOLD_STATUS,
        "g_minus1_pass": False,
        "reason": "Required author-approved payloads and detached approval sidecars are absent.",
        "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)},
        "server": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "project_root": str(project_root),
            "git_commit": git_commit,
            "git_worktree_dirty": bool(git_status),
        },
        "read_only_baseline_presence": {
            "v93_root": str(v93_root),
            "v93_root_exists": v93_root.is_dir(),
            "analysis_contract_path": str(v93_contract),
            "analysis_contract_sha256": sha256_file(v93_contract) if v93_contract.is_file() else None,
            "checksum_manifest_path": str(v93_manifest),
            "checksum_manifest_sha256": sha256_file(v93_manifest) if v93_manifest.is_file() else None,
        },
        "generated": {
            "schemas": sorted(schemas),
            "unsigned_payloads": sorted(templates),
            "jcs_test_vector_sha256": canonical_hash,
        },
        "required_initial_author_configs": list(INITIAL_AUTHOR_CONFIGS),
        "signed_initial_author_configs_found": [],
        "authorized_actions": ["AUTHOR_REVIEW_AND_SIGNATURE_ONLY"],
        "prohibited_actions": [
            "G0_SOURCE_NOISE_INVENTORY", "G05_POWER_SUPPORT", "G1_TO_G5_EXPLORATORY",
            "INJECTION", "PE", "CALIBRATION", "REAL_CATALOG_RERANK",
            "OVERWRITE_V93", "MODIFY_PAPER", "GIT_COMMIT_OR_PUSH",
        ],
    }
    write_json(output / "provenance/G_MINUS1_CONFIG_AUDIT.json", initial_audit)
    write_json(output / "STATUS.json", {
        "run_id": run_id,
        "status": HOLD_STATUS,
        "updated_at_utc": created,
        "next_required_action": "AUTHOR_SIGN_DESIGN_PE_INJECTION_AND_V93_CONTRACTS",
    })
    (output / HOLD_STATUS).write_text(created + "\n", encoding="ascii")

    (output / "RESULTS_INDEX_CN.md").write_text(build_readme(run_id), encoding="utf-8")
    (output / "reports/G_MINUS1_REPORT_CN.md").write_text(
        "# G-1 配置审计报告\n\n"
        f"- Run ID：`{run_id}`\n"
        f"- 状态：`{HOLD_STATUS}`\n"
        "- JSON Schema：已生成\n"
        "- 未签名 payload 与 sidecar：已生成\n"
        "- RFC 8785 canonicalization 测试向量：已生成并哈希\n"
        "- 作者签署配置：0/4\n"
        "- G0/G0.5/G1--G5：均未启动\n"
        "- 历史 v9.3：仅检查存在性和权威 manifest 哈希，未修改\n\n"
        "模板生成成功不等于 G-1 PASS。只有四份初始配置及其 detached sidecar "
        "通过结构、哈希和作者批准审计后，才可启动 G0。\n",
        encoding="utf-8",
    )
    (output / "reports/AUTHOR_ACTION_REQUIRED_CN.md").write_text(
        "# 作者下一步操作\n\n"
        "1. 复制 `contracts/unsigned_templates/` 到新的签署目录，不要原地覆盖模板。\n"
        "2. 填写 `DESIGN_CONFIG_DRAFT.json` 的 power 假设和资源上限。\n"
        "3. 填写并冻结 `PE_CONFIG.json` 的 waveform、prior、频段、PSD、sampler、DQ、calibration 和 map 约定。\n"
        "4. 填写 `INJECTION_AND_SELECTION_CONFIG.json` 的人口、响应、选择、阈值与 seeds。\n"
        "5. 解析并哈希 `V93_BASELINE_CONTRACT.json` 的三个模型、waveform/time 输入、逐 seed 权重和 strict manifest。\n"
        "6. 将四个 payload 标记为 `AUTHOR_APPROVED`，写明 `approved_for=[\"G0_AND_G05_ONLY\"]`。\n"
        "7. 重新计算每个 payload 的 RFC 8785 SHA-256，填写 detached `.approval.json`，再运行验证器。\n\n"
        "`ANALYSIS_CONFIG_FINAL.json` 当前不得签署；它只能在 G0.5 建议完成并由作者审核后签署。\n",
        encoding="utf-8",
    )

    # Every generated file is immutable input to the author-review package.
    manifest_rows = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and "package" not in p.parts):
        manifest_rows.append({
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_json(output / "provenance/G_MINUS1_FILE_MANIFEST.json", {
        "schema": "sky-background-v10-file-manifest-v1",
        "run_id": run_id,
        "files": manifest_rows,
    })
    return 20


if __name__ == "__main__":
    raise SystemExit(main())
