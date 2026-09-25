"""Compatibility API: C data, unchanged NEW-SCORE-ONLY training implementation."""
import argparse
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('legacy_unified_ab', Path(__file__).with_name('legacy_unified_ab.py'))
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
for name in dir(legacy):
    if not name.startswith('__'):
        globals()[name] = getattr(legacy, name)
ARMS = legacy.ARMS = ('C_PHYSICAL',)
# Keep the same model seeds for the controlled A/B/C comparison, not a new blind test.
SEEDS = legacy.SEEDS = (2026091721, 2026091722, 2026091723)
import c_data
legacy.worker_init = c_data.worker_init
legacy.generate_source = c_data.generate_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['generate', 'short'], required=True)
    parser.add_argument('--run', choices=RUNS, default='O3')
    parser.add_argument('--role', choices=['main', 'aux'], default='main')
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pilot', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'generate':
        legacy.generate(args.root, args.run, args.role, args.split, args.workers, args.pilot)
    else:
        legacy.train_short(args.root, args.run, args.arm, args.seed)


if __name__ == '__main__':
    main()
