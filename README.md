# LAWS-Lens

Paper-reproduction repository for **GWLR-UC-01 / C_PHYSICAL**, with the
paper's FPP/GPD analyses and DOMAIN04 runtime benchmark.

**The repository is public; the Zenodo data release is not yet published.**
The author authorized public visibility on 2026-09-25. License selection and
software/data contributor confirmation are pending; public visibility alone
does not grant a new open-source or data license. No scientific model, score,
ranking or manuscript was changed for this upload.

## Version identity

- Intended paper release: `v1.0.0` (not yet published).
- Verified data assembly: `r2`, corrected reproduction extension: `r2p1`.
- Paper snapshot: `afb86ff5c8f212aec579592f0d7439f3f909d931`.
- Exact eight data/environment archives: `release/RELEASE_PACKAGE_INDEX_R2.json`.
- Total indexed archive size: 8,641,841,028 bytes.
- Original validation receipts: `release/FINAL_VALIDATION_R2.json` and
  `release/DELIVERED_EXTENSION_REPLAY.json`.

Earlier names in frozen source paths identify provenance, not alternative
recommended results. Do not substitute A/B, PATH875 or earlier sky experiments
for the C_PHYSICAL paper baseline.

The archived dependency tree also retains older support scripts. Their presence
does not designate them as current entry points or authorize changing a paper.
Use only the finite reproduction commands below for release verification.

## Repository and data separation

This repository contains the frozen software dependency closure, configurations,
environment locks, plotting/replay entry points and release verification reports.
Trained weights, injection arrays, native sky maps, pair scores, manuscript
snapshot and the offline wheelhouse belong in the separately indexed archives.
They are intentionally not stored in Git history.

Zenodo upload is **incomplete, with large-file transport errors**. The local execution
network is restricted, but the release server can access Zenodo normally.
Unpublished drafts are [software](https://zenodo.org/deposit/22949904) and
[data/models](https://zenodo.org/deposit/22949906); sign in as the owner to view
them. They are not published or publicly downloadable releases.
`release/UPLOAD_STATUS.json` records the dated status. A checksum in a manifest
is not evidence that the corresponding file has already been uploaded.

The frozen software archive at commit `3356ccf` retains its pre-upload status
text. These newer upload notes and `READ_FIRST_ZENODO_DRAFT.md` supersede that
historical status without changing the archived scientific payload.

## Reproduction

1. Obtain the eight exact archives listed in the release index from the author
   until the Zenodo draft/release is available. Do not use the superseded `r2`
   extension; use `r2p1`.
2. Check their SHA-256 values. Extract the four modular r1 archives and two r2
   additions into a **new empty release directory**. Extract native maps and the
   wheelhouse separately. Keep this Git checkout and that data assembly separate.
3. Use CPython 3.12 on Linux x86_64. The frozen environment uses CUDA 12.8 wheels;
   reproducing GPU checks requires compatible NVIDIA hardware and drivers.
   The hash-locked requirements are under
   `supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/environment/`.
4. Run `scripts/create_environment.py --help` for the offline wheelhouse installer.
   It creates a new environment; it never modifies an existing environment.
5. In the extracted release directory, using that environment's Python:

```bash
python -B scripts/reproduce.py paper --release "$PWD"
python -B scripts/reproduce.py features --release "$PWD"
python -B scripts/reproduce.py models --release "$PWD"
python -B scripts/reproduce.py injections --release "$PWD"
python -B scripts/reproduce.py metrics-sky --release "$PWD"
python -B extension/scripts/reproduce_extension.py all --release "$PWD"
```

Read `extension/README_CN.md` from the assembled archives for the detailed
validation scope. Prefer these wrappers to running archived training scripts
directly: historical scripts retain their original paths, and wrappers provide
the isolated release mapping. Recovery validation needs about 12 GiB of free
workspace for temporary products.

The checkout alone supports a dependency-free integrity check:

```bash
python3 tools/verify_software.py
```

## Verified scope and limits

- 847 original input files were hash-checked; only three pinned public GW-LMC
  tables were freshly downloaded, not the whole public-input corpus.
- Frozen validation/test/real model inference was reproduced; this is not a new
  training run or a fresh blind scientific test.
- All 500 fixed subcatalogs were recomputed: 288,000 scalar comparisons passed.
  Overlapping subcatalogs are not independent experiments.
- Six representative events passed noisy-strain generation, full 8,192-template
  recovery and BAYESTAR replay, including exact native MOC pixel identifiers.
  This is not a whole-population regeneration claim.
- Two original figure scripts and eleven explicitly labelled data-equivalent
  reconstructions were checked. The missing original layouts were not recovered.
- The archive extraction replay passed without changing the frozen payload.
- No independent second physical host was validated.

Empirical conditional FPP is not a lensing posterior probability. GPD endpoint
`B=4` is a modelling assumption, with `B=4.5/5` sensitivity material retained.
The release does not establish effective exposure for an annual FAR. Public
Hanabi-table overlap is not a new Hanabi analysis or a lensing confirmation.

## Licensing and publication

See `LICENSE_STATUS.md`. No repository-wide license has been granted here.
Public visibility is not a certification of third-party redistribution rights.
Zenodo publication must wait for complete verified uploads, author-confirmed
metadata and appropriate licenses/third-party rights review.

Chinese delivery notes: `release/FINAL_DELIVERY_R2_CN.md`.
