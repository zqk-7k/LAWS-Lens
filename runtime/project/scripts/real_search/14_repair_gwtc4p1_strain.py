from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "gwtc4p1_data_completion_20260628"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hdf5_opens(path: Path) -> tuple[bool, str]:
    try:
        with h5py.File(path, "r") as h5:
            _ = list(h5.keys())
        return True, "open_ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def file_state(row: pd.Series, repo_root: Path) -> dict[str, Any]:
    path = repo_root / str(row["local_path"])
    expected = pd.to_numeric(pd.Series([row.get("expected_bytes")]), errors="coerce").iloc[0]
    expected_int = None if pd.isna(expected) else int(expected)
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    size_ok = exists and size > 0 and (expected_int is None or size == expected_int)
    open_ok = False
    open_status = "not_checked"
    if size_ok:
        open_ok, open_status = hdf5_opens(path)
    else:
        open_status = "missing_or_size_mismatch"
    return {
        "event_name": row.get("event_name", ""),
        "detector": row.get("detector", ""),
        "path": str(path),
        "url": row.get("url", ""),
        "expected_bytes": expected_int,
        "exists": bool(exists),
        "local_bytes": int(size),
        "size_ok": bool(size_ok),
        "open_ok": bool(open_ok),
        "open_status": open_status,
        "ok": bool(size_ok and open_ok),
    }


def download_once(url: str, tmp_path: Path, timeout: int) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "gw-catalog-gwtc4p1-strain-repair/1.0"})
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=timeout) as response, tmp_path.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=1024 * 1024)
    return tmp_path.stat().st_size


def repair_one(row: dict[str, Any], repo_root: Path, max_attempts: int, timeout: int, sleep_s: float) -> dict[str, Any]:
    local = repo_root / str(row["local_path"])
    tmp = local.with_name(local.name + ".repair.part")
    expected = row.get("expected_bytes")
    expected_int = int(expected) if str(expected).strip().isdigit() else None
    result: dict[str, Any] = {
        "event_name": row.get("event_name", ""),
        "detector": row.get("detector", ""),
        "local_path": str(local),
        "url": row.get("url", ""),
        "expected_bytes": expected_int,
        "started_at_utc": utc_now(),
        "status": "not_started",
        "attempts": 0,
        "tmp_path": str(tmp),
    }
    for attempt in range(1, max_attempts + 1):
        result["attempts"] = attempt
        try:
            n = download_once(str(row["url"]), tmp, timeout=timeout)
            result["tmp_bytes"] = int(n)
            if expected_int is not None and n != expected_int:
                result["status"] = "retry_size_mismatch"
                result["message"] = f"tmp_bytes={n}, expected={expected_int}"
                time.sleep(sleep_s)
                continue
            open_ok, open_status = hdf5_opens(tmp)
            result["tmp_open_status"] = open_status
            if not open_ok:
                result["status"] = "retry_hdf5_open_failed"
                result["message"] = open_status
                time.sleep(sleep_s)
                continue
            os.replace(tmp, local)
            final_ok, final_status = hdf5_opens(local)
            final_size = local.stat().st_size if local.exists() else 0
            result.update({
                "status": "repaired" if final_ok else "failed_final_open",
                "finished_at_utc": utc_now(),
                "final_bytes": int(final_size),
                "final_open_status": final_status,
            })
            return result
        except Exception as exc:
            result["status"] = "retry_exception" if attempt < max_attempts else "failed_exception"
            result["message"] = f"{type(exc).__name__}:{exc}"
            time.sleep(sleep_s)
    result["finished_at_utc"] = utc_now()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep-s", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-bad-first", action="store_true", help="Remove confirmed bad local strain files before redownloading them.")
    args = parser.parse_args()

    run_dir = args.run_dir
    data_dir = run_dir / "data"
    logs_dir = run_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "strain_manifest_gwtc4p1.csv"
    manifest = pd.read_csv(manifest_path)
    states = [file_state(row, REPO_ROOT) for _, row in manifest.iterrows()]
    states_df = pd.DataFrame(states)
    before_path = data_dir / "gwtc4p1_strain_integrity_before_repair_20260629.csv"
    states_df.to_csv(before_path, index=False)
    bad_paths = set(states_df.loc[~states_df["ok"], "path"])
    todo = []
    for _, row in manifest.iterrows():
        local = str(REPO_ROOT / str(row["local_path"]))
        if local in bad_paths:
            todo.append(row.to_dict())

    summary = {
        "generated_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "target_manifest": str(manifest_path),
        "total_manifest_rows": int(len(manifest)),
        "ok_before": int(states_df["ok"].sum()),
        "bad_before": int((~states_df["ok"]).sum()),
        "dry_run": bool(args.dry_run),
        "delete_bad_first": bool(args.delete_bad_first),
        "max_workers": args.max_workers,
        "max_attempts": args.max_attempts,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        (data_dir / "gwtc4p1_strain_repair_dry_run_20260629.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    deleted_bad: list[dict[str, Any]] = []
    if args.delete_bad_first:
        for state in states:
            if state["ok"]:
                continue
            path = Path(state["path"])
            if path.exists() and path.is_file():
                size = path.stat().st_size
                path.unlink()
                deleted_bad.append({
                    "event_name": state["event_name"],
                    "detector": state["detector"],
                    "path": str(path),
                    "deleted_bytes": int(size),
                    "reason": state["open_status"],
                })
        (data_dir / "gwtc4p1_deleted_bad_strain_files_20260629.json").write_text(
            json.dumps({
                "generated_at_utc": utc_now(),
                "deleted_count": len(deleted_bad),
                "deleted_bytes": int(sum(row["deleted_bytes"] for row in deleted_bad)),
                "files": deleted_bad,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    log_path = logs_dir / "gwtc4p1_strain_repair_20260629.jsonl"
    lock = threading.Lock()
    repaired = 0
    failed = 0
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log_fh:
        with futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_map = {
                pool.submit(repair_one, row, REPO_ROOT, args.max_attempts, args.timeout, args.sleep_s): row
                for row in todo
            }
            for idx, fut in enumerate(futures.as_completed(future_map), start=1):
                result = fut.result()
                if result.get("status") == "repaired":
                    repaired += 1
                else:
                    failed += 1
                with lock:
                    log_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    log_fh.flush()
                elapsed = time.time() - started
                print(
                    f"[{idx}/{len(todo)}] status={result.get('status')} repaired={repaired} failed={failed} "
                    f"elapsed_s={elapsed:.1f} event={result.get('event_name')} det={result.get('detector')}",
                    flush=True,
                )

    after_states = [file_state(row, REPO_ROOT) for _, row in manifest.iterrows()]
    after_df = pd.DataFrame(after_states)
    after_path = data_dir / "gwtc4p1_strain_integrity_after_repair_20260629.csv"
    after_df.to_csv(after_path, index=False)
    final_summary = {
        **summary,
        "finished_at_utc": utc_now(),
        "attempted_repairs": int(len(todo)),
        "repaired_count": int(repaired),
        "failed_count": int(failed),
        "ok_after": int(after_df["ok"].sum()),
        "bad_after": int((~after_df["ok"]).sum()),
        "repair_log": str(log_path),
        "before_integrity_csv": str(before_path),
        "after_integrity_csv": str(after_path),
        "bad_after_sample": after_df.loc[~after_df["ok"], ["event_name", "detector", "path", "local_bytes", "expected_bytes", "open_status"]].head(20).to_dict("records"),
    }
    (data_dir / "gwtc4p1_strain_repair_summary_20260629.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
