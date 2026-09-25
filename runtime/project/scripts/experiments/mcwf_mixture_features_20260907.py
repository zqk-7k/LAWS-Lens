#!/usr/bin/env python3
"""Expand independently generated waveform features; no historical writes."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mcwf_dense_features_20260907 as dense


def run(root):
    for dep in ('gwtc3', 'gwtc4'):
        noise = root / f'expanded_data/{dep}/noise'
        for split in ('validation', 'train'):
            src = root / f'expanded_data/{dep}/{split}'
            if not (src / 'COMPLETE.json').exists():
                raise RuntimeError('Independent waveform data incomplete')
            meta = pd.read_parquet(src / 'event_metadata.parquet')
            dense.grouped(root, np.load(src / 'raw2s.npy', mmap_mode='r'),
                np.load(noise / 'frequency.npy'), np.load(noise / 'psd.npy', mmap_mode='r'),
                meta.noise_bank_index.to_numpy(int), root / f'mixture_density/features/{dep}/{split}_extra.npy')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
