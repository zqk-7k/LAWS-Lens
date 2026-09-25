#!/usr/bin/env python3
"""Validate detached author approvals without filling or signing any config."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


INITIAL = (
    "DESIGN_CONFIG_DRAFT",
    "PE_CONFIG",
    "INJECTION_AND_SELECTION_CONFIG",
    "V93_BASELINE_CONTRACT",
)


def load_initializer(path: Path):
    spec = importlib.util.spec_from_file_location("sky_v10_initializer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(has_null(item) for item in value.values())
    if isinstance(value, list):
        return any(has_null(item) for item in value)
    return False


def schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate the deliberately constrained JSON-Schema subset in this run."""
    errors: list[str] = []
    if "anyOf" in schema:
        branches = [schema_errors(value, branch, path) for branch in schema["anyOf"]]
        if not any(not branch for branch in branches):
            errors.append(f"{path}:anyOf_failed")
        return errors
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}:const_mismatch")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}:not_in_enum")

    expected = schema.get("type")
    type_ok = True
    if expected == "object":
        type_ok = isinstance(value, dict)
    elif expected == "array":
        type_ok = isinstance(value, list)
    elif expected == "string":
        type_ok = isinstance(value, str)
    elif expected == "integer":
        type_ok = isinstance(value, int) and not isinstance(value, bool)
    elif expected == "number":
        type_ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected == "boolean":
        type_ok = isinstance(value, bool)
    elif expected == "null":
        type_ok = value is None
    if expected and not type_ok:
        errors.append(f"{path}:expected_{expected}")
        return errors

    if isinstance(value, dict):
        required = schema.get("required", [])
        errors.extend(f"{path}.{key}:required" for key in required if key not in value)
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{key}:additional_property" for key in value if key not in properties)
        for key, child in properties.items():
            if key in value:
                errors.extend(schema_errors(value[key], child, f"{path}.{key}"))
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            errors.append(f"{path}:minProperties")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}:minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}:maxItems")
        if schema.get("uniqueItems"):
            serialized = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(serialized) != len(set(serialized)):
                errors.append(f"{path}:uniqueItems")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, schema["items"], f"{path}[{index}]"))

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}:minLength")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            errors.append(f"{path}:pattern")
        if schema.get("format") == "date-time":
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"{path}:invalid_date_time")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}:minimum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}:exclusiveMinimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}:maximum")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--signed-config-dir", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    signed = args.signed_config_dir.resolve()
    initializer = load_initializer(run_root / "scripts/00a_generate_json_schemas_and_unsigned_templates_then_hold.py")
    rows = []
    all_ok = True
    for kind in INITIAL:
        payload_path = signed / f"{kind}.json"
        sidecar_path = signed / f"{kind}.approval.json"
        errors = []
        payload = None
        sidecar = None
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"payload_unreadable:{exc}")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"sidecar_unreadable:{exc}")
        if payload is not None:
            schema_path = run_root / f"contracts/schemas/{kind}.schema.json"
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                errors.extend(schema_errors(payload, schema))
            except Exception as exc:
                errors.append(f"schema_unreadable:{exc}")
            if payload.get("config_kind") != kind:
                errors.append("config_kind_mismatch")
            if payload.get("lifecycle_status") != "AUTHOR_APPROVED":
                errors.append("payload_not_author_approved")
            if payload.get("approved_for") != ["G0_AND_G05_ONLY"]:
                errors.append("payload_approved_for_must_equal_G0_AND_G05_ONLY")
            if has_null(payload):
                errors.append("payload_contains_null")
        if payload is not None and sidecar is not None:
            try:
                sidecar_schema = json.loads(
                    (run_root / "contracts/schemas/APPROVAL_SIDECAR.schema.json").read_text(encoding="utf-8")
                )
                errors.extend(schema_errors(sidecar, sidecar_schema, "$.approval"))
            except Exception as exc:
                errors.append(f"sidecar_schema_unreadable:{exc}")
            digest = hashlib.sha256(initializer.jcs_bytes(payload)).hexdigest()
            if sidecar.get("payload_filename") != payload_path.name:
                errors.append("sidecar_filename_mismatch")
            if sidecar.get("payload_sha256_jcs") != digest:
                errors.append("sidecar_hash_mismatch")
            if sidecar.get("approval_status") != "AUTHOR_APPROVED":
                errors.append("sidecar_not_author_approved")
            if sidecar.get("approved_for") != ["G0_AND_G05_ONLY"]:
                errors.append("sidecar_approved_for_must_equal_G0_AND_G05_ONLY")
            if not sidecar.get("author_identity"):
                errors.append("author_identity_missing")
            if sidecar.get("signature_method") == "UNSIGNED" or not sidecar.get("signature"):
                errors.append("signature_missing")
            try:
                datetime.fromisoformat(str(sidecar.get("approved_at_utc", "")).replace("Z", "+00:00"))
            except ValueError:
                errors.append("approved_at_utc_invalid")
        ok = not errors
        all_ok &= ok
        rows.append({"config_kind": kind, "passed": ok, "errors": errors})

    status = "PASS_G_MINUS1_AUTHOR_CONFIGS" if all_ok else "HOLD_FOR_AUTHOR_SIGNATURE"
    audit = {"schema": "sky-background-v10-signed-config-audit-v1", "status": status, "checks": rows}
    target = run_root / "provenance/G_MINUS1_SIGNED_CONFIG_VALIDATION.json"
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if all_ok else 20


if __name__ == "__main__":
    raise SystemExit(main())
