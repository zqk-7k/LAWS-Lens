#!/usr/bin/env python3
"""Prebuild v6 compact and multi-noise training arrays without model access."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
V5_ROOT = REPO / "results/real_noise_injection_v5_physical_source_20260721"
DEFAULT_OUT = REPO / "results/real_noise_injection_v6_unified_physics_20260721"


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = module_from(
    REPO / "scripts/experiments/20_real_noise_injection_v3_physical.py",
    "v3_for_v6_prebuild",
)
v5 = module_from(
    REPO / "scripts/real_search/35_real_noise_injection_v5_physical_source.py",
    "v5_for_v6_prebuild",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--deployments", nargs="+", default=["GWTC3", "GWTC4"])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--variants", type=int, default=8)
    args = parser.parse_args()

    for deployment in args.deployments:
        shared = V5_ROOT / deployment.lower() / "shared"
        v3.ensure_link(args.out_root / deployment.lower() / "shared", shared)
        source_bank = shared / "physical_h1l1_source_bank"
        for seed in args.seeds:
            seed_dir = args.out_root / deployment.lower() / f"seed_{seed}"
            v3.prepare_seed_layout(seed_dir, v3.SOURCES[deployment])
            v5.materialize_dataset_v5(
                seed_dir,
                shared,
                deployment,
                seed + 100,
                args.samples,
            )
            for family in ("SIS", "PM"):
                marker = (
                    seed_dir
                    / "data/real_noise_injections"
                    / f"multinoise_{family.lower()}_train_v{args.variants}"
                    / "multinoise_summary.json"
                )
                if marker.exists():
                    continue
                subprocess.run(
                    [
                        sys.executable,
                        str(REPO / "scripts/real_search/36_materialize_multinoise_v5.py"),
                        "--seed-root",
                        str(seed_dir),
                        "--source-bank",
                        str(source_bank),
                        "--family",
                        family,
                        "--seed",
                        str(seed),
                        "--samples",
                        str(args.samples),
                        "--variants-per-source",
                        str(args.variants),
                    ],
                    cwd=REPO,
                    check=True,
                )
            print(f"complete {deployment} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
