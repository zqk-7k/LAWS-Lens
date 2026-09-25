from __future__ import annotations

import io
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.gwtc import config


def get_json(url: str, **params: Any) -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        response = requests.get(url, params=params or None, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        return True, response.json(), None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def get_text(url: str) -> tuple[bool, str | None, str | None]:
    try:
        response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        return True, response.text, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    norm_to_original = {normalize_column(col): col for col in columns}
    for candidate in candidates:
        if candidate in norm_to_original:
            return norm_to_original[candidate]
    for col in columns:
        normalized = normalize_column(col)
        if any(candidate in normalized for candidate in candidates):
            return col
    return None


def probe_gwosc_event_api() -> dict[str, Any]:
    ok, text, error = get_text(config.GWOSC_ALLEVENTS_CSV)
    result: dict[str, Any] = {
        "url": config.GWOSC_ALLEVENTS_CSV,
        "reachable": ok,
        "error": error,
        "row_count": 0,
        "columns": [],
        "detected_columns": {},
        "catalog_counts": {},
        "gwtc5_or_o4b_count": 0,
    }
    if not ok or text is None:
        return result

    df = pd.read_csv(io.StringIO(text))
    columns = list(df.columns)
    result["row_count"] = int(len(df))
    result["columns"] = columns

    detected = {
        "event_name": find_column(columns, ["commonname", "event_name", "name", "superevent", "id"]),
        "gps_time": find_column(columns, ["gps", "gps_time", "gpstime", "tc"]),
        "network_snr": find_column(columns, ["network_snr", "snr", "matched_filter_snr"]),
        "ra": find_column(columns, ["ra", "right_ascension"]),
        "dec": find_column(columns, ["dec", "declination"]),
        "chirp_mass": find_column(columns, ["chirp_mass", "mchirp", "mass_1_source"]),
        "catalog": find_column(columns, ["catalog", "catalog_shortname", "catalogs", "version"]),
    }
    result["detected_columns"] = detected

    catalog_col = detected.get("catalog")
    if catalog_col:
        counts = Counter()
        for value in df[catalog_col].fillna("").astype(str):
            if not value:
                counts[""] += 1
            else:
                for token in re.split(r"[,;| ]+", value):
                    token = token.strip()
                    if token:
                        counts[token] += 1
        result["catalog_counts"] = dict(counts.most_common())
        mask = df[catalog_col].fillna("").astype(str).str.contains("GWTC-5|GWTC5|O4b|O4", case=False, regex=True)
        result["gwtc5_or_o4b_count"] = int(mask.sum())
    else:
        any_text = df.astype(str).agg(" ".join, axis=1)
        result["gwtc5_or_o4b_count"] = int(any_text.str.contains("GWTC-5|GWTC5|O4b|O4", case=False, regex=True).sum())

    required = ["event_name", "gps_time", "network_snr"]
    result["has_required_core_columns"] = all(detected.get(key) for key in required)
    result["has_sky_or_mass_columns"] = any(detected.get(key) for key in ["ra", "dec", "chirp_mass"])
    return result


def zenodo_files(record: dict[str, Any]) -> list[dict[str, Any]]:
    files = []
    for item in record.get("files", []):
        key = item.get("key") or item.get("filename") or ""
        links = item.get("links", {})
        files.append({
            "key": key,
            "size": item.get("size"),
            "download": links.get("self") or links.get("download"),
        })
    return files


def probe_zenodo_record(url: str, patterns: dict[str, str]) -> dict[str, Any]:
    ok, payload, error = get_json(url)
    result: dict[str, Any] = {
        "url": url,
        "reachable": ok,
        "error": error,
        "record_id": None,
        "title": None,
        "file_count": 0,
        "patterns": {},
        "sample_files": [],
    }
    if not ok or payload is None:
        return result
    files = zenodo_files(payload)
    result["record_id"] = payload.get("id")
    result["title"] = payload.get("metadata", {}).get("title")
    result["file_count"] = len(files)
    result["sample_files"] = [f["key"] for f in files[:20]]
    for label, pattern in patterns.items():
        rx = re.compile(pattern)
        matches = [f["key"] for f in files if rx.search(f["key"])]
        result["patterns"][label] = {
            "pattern": pattern,
            "count": len(matches),
            "sample": matches[:20],
        }
    return result


def probe_gwtc5_pe_search() -> dict[str, Any]:
    ok, payload, error = get_json(config.ZENODO_GWTC5_PE_SEARCH)
    result: dict[str, Any] = {
        "url": config.ZENODO_GWTC5_PE_SEARCH,
        "reachable": ok,
        "error": error,
        "hits": [],
        "full_pe_candidate_found": False,
    }
    if not ok or payload is None:
        return result
    hits = payload.get("hits", {}).get("hits", [])
    for hit in hits[:10]:
        metadata = hit.get("metadata", {})
        title = metadata.get("title", "")
        files = [f.get("key", "") for f in hit.get("files", [])[:10]]
        full_pe_hint = bool(re.search(r"parameter|posterior|PEDataRelease|sample", title + " " + " ".join(files), re.I))
        result["hits"].append({
            "id": hit.get("id"),
            "title": title,
            "created": hit.get("created"),
            "file_count": len(hit.get("files", [])),
            "sample_files": files,
            "full_pe_hint": full_pe_hint,
        })
        if full_pe_hint:
            result["full_pe_candidate_found"] = True
    return result


def decide_capabilities(status: dict[str, Any]) -> dict[str, Any]:
    gwosc = status["gwosc_event_api"]
    gwtc3 = status["zenodo_gwtc3"]
    gwtc21 = status["zenodo_gwtc21"]
    gwtc5 = status["zenodo_gwtc5_candidate_bayestar"]
    gwtc5_search = status["zenodo_gwtc5_pe_search"]

    gwtc3_pe = gwtc3["reachable"] and (
        gwtc3["patterns"].get("gwtc3_mixed_nocosmo_h5", {}).get("count", 0) > 0
        or gwtc3["patterns"].get("gwtc3_mixed_cosmo_h5", {}).get("count", 0) > 0
    )
    gwtc3_sky = gwtc3["patterns"].get("gwtc3_skymap_tar", {}).get("count", 0) > 0
    gwtc21_pe = gwtc21["reachable"] and gwtc21["file_count"] > 0
    gwtc5_observables = bool(gwosc["reachable"] and gwosc.get("has_required_core_columns")) or gwtc5["reachable"]
    gwtc5_full = bool(gwtc5_search.get("full_pe_candidate_found"))

    levels = []
    if gwtc3_pe and gwtc21_pe:
        levels.append("FULL_GWTC3")
    if gwtc5_observables:
        levels.append("GWTC5_OBSERVABLES")
    if gwtc5_full:
        levels.append("GWTC5_FULL")

    recommended = "FULL_GWTC3 first"
    if "GWTC5_OBSERVABLES" in levels:
        recommended += "; then GWTC5 observable-only scalability"
    if "GWTC5_FULL" in levels:
        recommended += "; optionally GWTC5 full PE"

    return {
        "levels": levels,
        "FULL_GWTC3": "FULL_GWTC3" in levels,
        "GWTC5_OBSERVABLES": "GWTC5_OBSERVABLES" in levels,
        "GWTC5_FULL": "GWTC5_FULL" in levels,
        "gwtc3_full_pe": bool(gwtc3_pe and gwtc3_sky and gwtc21_pe),
        "gwtc5_observables": bool(gwtc5_observables),
        "gwtc5_full_pe": bool(gwtc5_full),
        "recommended_path": recommended,
    }


def main() -> None:
    config.SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gwosc_event_api": probe_gwosc_event_api(),
        "zenodo_gwtc3": probe_zenodo_record(config.ZENODO_GWTC3_RECORD, {
            "gwtc3_mixed_nocosmo_h5": r"IGWN-GWTC3p0-v2-GW.*_PEDataRelease_mixed_nocosmo\.h5$",
            "gwtc3_mixed_cosmo_h5": r"IGWN-GWTC3p0-v2-GW.*_PEDataRelease_mixed_cosmo\.h5$",
            "gwtc3_skymap_tar": r"IGWN-GWTC3p0-v2-PESkyLocalizations\.tar\.gz$",
        }),
        "zenodo_gwtc21": probe_zenodo_record(config.ZENODO_GWTC21_RECORD, {
            "gwtc21_h5": r"\.h5$",
            "gwtc21_skymap": r"Sky|sky|fits|tar",
            "gw170104": r"GW170104",
            "gw170814": r"GW170814",
        }),
        "zenodo_gwtc5_candidate_bayestar": probe_zenodo_record(config.ZENODO_GWTC5_CANDIDATE_RECORD, {
            "fits_or_fits_gz": r"\.fits(\.gz)?$",
            "bayestar": r"Bayestar|bayestar|BAYESTAR",
            "csv": r"\.csv$",
        }),
        "zenodo_gwtc5_pe_search": probe_gwtc5_pe_search(),
    }
    status["capabilities"] = decide_capabilities(status)
    config.SOURCE_STATUS_JSON.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    cap = status["capabilities"]
    print(
        "GWTC-3 full PE = {gwtc3}; GWTC-5 observables = {gwtc5obs}; "
        "GWTC-5 full PE = {gwtc5full}; recommended path = {path}".format(
            gwtc3="yes" if cap["gwtc3_full_pe"] else "no",
            gwtc5obs="yes" if cap["gwtc5_observables"] else "no",
            gwtc5full="yes" if cap["gwtc5_full_pe"] else "no",
            path=cap["recommended_path"],
        ),
        flush=True,
    )
    print(f"Wrote {config.SOURCE_STATUS_JSON}", flush=True)


if __name__ == "__main__":
    main()
