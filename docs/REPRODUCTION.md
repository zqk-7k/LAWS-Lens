# Reproducing the analysis

## Download and assemble

Download these seven complete archives from the
[published data record](https://doi.org/10.5281/zenodo.22949906):

| Archive | Role |
| --- | --- |
| LAWS_Lens_v1_DRAFT_code_models_scores_20260924.tar.gz | Code, weights, calibration and scores |
| LAWS_Lens_v1_DRAFT_paper_fpp_speed_20260924.tar.gz | Original manuscript snapshot, background and runtime analysis |
| LAWS_Lens_v1_DRAFT_validation_fixtures_20260924.tar.gz | Validation inputs |
| LAWS_Lens_v1_DRAFT_release_audit_v2_20260924.tar.gz | Manifests and reports |
| LAWS_Lens_v1_DRAFT_reproduction_extension_r2p1_20260924.tar.gz | Reproduction extension |
| LAWS_Lens_v1_DRAFT_test_real_fixtures_r2_20260924.tar.gz | Test and observed-event inputs |
| GWLR_UC_01_native_BAYESTAR_maps.tar.gz | Native injection sky maps |

Check SHA-256 against `release/PUBLISHED_PACKAGES.json`. Extract the first six
into a new empty release directory. Keep the map archive separate and follow
the map location in the extracted extension instructions. Use r2p1, not the
superseded r2 extension; do not extract duplicate transport fragments.

## Environment

The recorded environment is Linux x86_64, Python 3.12.3 and PyTorch 2.8.0 with
CUDA 12.8. GPU inference needs a compatible NVIDIA driver. Versions and wheel
hashes are in `supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/environment/`.

The historical offline wheelhouse is not publicly distributed. Dependencies
can be obtained from PyPI and the PyTorch CUDA repository. In a new Linux
Python 3.12 environment, use:

```bash
python3.12 -m venv laws-env
source laws-env/bin/activate
python -m pip install --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/environment/requirements.linux-x86_64-cp312-cu128.lock
python -m pip check
```

If a matching wheel is unavailable, report its package/version rather than
silently substituting dependencies. The recorded installation checks used
offline wheels; the complete online installation route remains to be tested.

## Analysis checks

From the extracted release directory, using the configured environment:

```bash
python -B scripts/reproduce.py paper --release "$PWD"
python -B scripts/reproduce.py features --release "$PWD"
python -B scripts/reproduce.py models --release "$PWD"
python -B scripts/reproduce.py injections --release "$PWD"
python -B scripts/reproduce.py metrics-sky --release "$PWD"
python -B extension/scripts/reproduce_extension.py all --release "$PWD"
```

The wrappers resolve archived paths and write per-stage logs. Representative
recovery checks need about 12 GiB of temporary workspace. The assembled
`extension/README_CN.md` describes the fixtures.

The recorded validation covers pair metrics and 500 fixed subsets, features,
inference from released models, and six representative injection/recovery cases.
It includes 288,000 scalar metric comparisons. Fresh training and full-population
regeneration are separate computations. Later sources are mapped in VERSIONS.md.
