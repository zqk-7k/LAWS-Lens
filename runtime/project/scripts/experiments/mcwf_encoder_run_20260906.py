#!/usr/bin/env python3
"""Resumable finite encoder ablations; no automatic success/adoption."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mcwf_encoder_body_20260906 as body
import mcwf_development_20260905 as dev


def run(root, configurations):
    body.initialize(root)
    scripts = Path(__file__).resolve().parent
    for name in ("mcwf_encoder_body_20260906.py", "mcwf_encoder_evaluate_20260906.py", Path(__file__).name):
        shutil.copy2(scripts / name, root / "scripts" / name)
    for config in configurations:
        for dep in ("gwtc3", "gwtc4"):
            if config == "O3-RUNMATCHED-RAW" and dep != "gwtc3":
                continue
            for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
                for action in ("train", "evaluate"):
                    script = scripts / ("mcwf_encoder_body_20260906.py" if action == "train" else "mcwf_encoder_evaluate_20260906.py")
                    cmd = [sys.executable, "-u", "-B", str(script), "--root", str(root), "--action", action,
                           "--config", config, "--deployment", dep, "--seed", str(ms)]
                    if action == "evaluate":
                        cmd += ["--eval-seed", str(es)]
                    started = datetime.now(timezone.utc).isoformat()
                    log = root / "logs" / f"{config}_{dep}_{ms}_{action}.log"
                    print(json.dumps({"start": started, "config": config, "dep": dep, "seed": ms, "action": action}), flush=True)
                    with log.open("a") as stream:
                        status = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, cwd=dev.PROJECT).returncode
                    if status:
                        dev.json_write(root / "contracts/EXECUTION_ERROR.json", {"command": cmd, "exit_code": status, "log": str(log), "goal_achieved": False})
                        raise RuntimeError(f"Execution failed: {log}")
        subprocess.run([sys.executable, "-u", "-B", str(scripts / "mcwf_encoder_evaluate_20260906.py"),
                        "--root", str(root), "--action", "summarize"], check=True)
    dev.json_write(root / "contracts/ABLATION_COMPUTE_COMPLETE.json", {"configurations": configurations,
                   "goal_achieved": False, "status": "AWAITING_SCIENTIFIC_GUARDRAIL_AUDIT_NOT_FINAL_SUCCESS"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--configs", nargs="+", choices=body.CONFIGS, default=list(body.CONFIGS))
    args = p.parse_args()
    run(args.root, args.configs)
