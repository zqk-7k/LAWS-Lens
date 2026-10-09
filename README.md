# LAWS-Lens

LAWS-Lens (Low-latency AI-based Waveform Search for gravitational-wave lensing)
ranks candidate lensed event pairs using detector strain, event times and sky
localization maps. Four neural waveform branches compare morphology and predicted
source parameters. Their calibrated score is combined with time-delay and sky-overlap
scores. Full mass and spin posteriors are used after ranking, not as ranking inputs.

## Software and data

- [Software v1.0.0-rc2](https://doi.org/10.5281/zenodo.22949904)
- [Data and trained models v1.0.0-rc2](https://doi.org/10.5281/zenodo.22949906)
- [Download and reproduction instructions](docs/REPRODUCTION.md)
- [Code entry points](docs/CODE_MAP.md)
- [Archive and manuscript version map](docs/VERSIONS.md)

Both Zenodo records are public. Download complete archives, not transport fragments.
Filenames retain their original DRAFT labels to preserve the published checksums.
The software record also contains rc1; use rc2 for this release.

The primary experiment fits separate models for O3, O4a and O4b. Each run uses
three random seeds with fixed source and noise partitions. The archived identifier
GWLR-UC-01 / C_PHYSICAL links the implementation to its models and scores.

## Repository layout

| Directory | Contents |
| --- | --- |
| runtime/training/ | Injection generation, recovery and waveform training |
| runtime/completion/ | Calibration, evaluation and observed-event scoring |
| runtime/project/ | Shared modules and retained development scripts |
| scripts/ | Portable entry points for the extracted release |
| extension/ | Additional input and inference checks |
| supplement/ | Environment specifications and input provenance |
| paper/ | Updated figure and table source data |
| release/ | Archive manifests and dated validation records |
| uploaded_snapshot/, speed/ | Earlier source snapshots and runtime analysis |

The data archives supply weights, calibrations, strain fixtures, maps and scores.
They are separate from this Git checkout. Historical scripts retain original paths;
the reproduction wrappers resolve them within the assembled release.

## Quick integrity check

This command uses only the Python standard library:

```bash
python tools/verify_software.py
```

For inference and injection regeneration, follow the Linux/Python 3.12 setup in
[REPRODUCTION.md](docs/REPRODUCTION.md).

## Interpreting results

Real-event lists use mean rank across three models. Displayed scores and channel
contributions are model means and can be non-monotonic in rank. Injection recall
measures recovery of known lensed images; real GWTC lensing labels are unknown.

Empirical FPP measures score exceedances in the specified simulated background.
GPD estimates assume an upper bound of 4, with bounds of 4.5 and 5 used for
sensitivity analysis. Neither quantity is a lensing posterior probability or an
annual false-alarm rate.

## Citation and licence

Use [CITATION.cff](CITATION.cff) and the versioned software and data DOIs.
Original software is MIT licensed. Self-generated data and model weights use
CC BY 4.0. Third-party materials retain their original terms; see
[LICENSE_STATUS.md](LICENSE_STATUS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
