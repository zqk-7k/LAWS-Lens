# Third-party materials

The root MIT license applies to original LAWS-Lens software, not to ownership
of upstream software, public inputs, manuscript templates or reference material.
The author-approved CC BY 4.0 grant applies only to self-generated data and model
weights. Neither grant changes third-party terms.

## Phazap

The archived speed-benchmark tree contains eight Phazap Python files under:

`speed/trilens_speed_domain04_20260917T155616Z/scripts/archived_inputs/results/lensrank_speed_scientific_pilot_20260917T073200Z/vendor/site/phazap/`

Upstream: https://github.com/ezquiaga/phazap

Copyright (c) 2023 Jose Maria Ezquiaga. Phazap is MIT-licensed; the exact upstream
copyright and permission text is reproduced in `licenses/Phazap-MIT.txt`.
The frozen Python files are unchanged. This notice repairs an omitted attribution;
it does not claim that the archived code is the latest upstream release.

## Dependencies and public scientific inputs

The three GW-LMC input tables use upstream commit
`55c9e1df770e4ba21815fd20233eab240e551d9d`, whose CC0-1.0 license is retained
in `licenses/GW-LMC-CC0-1.0.txt`. They are not original LAWS-Lens CC BY data.
Repository: https://github.com/LensedGW/GW-LMC

LALSuite, ligo.skymap, Bilby, PyTorch, NumPy, SciPy and other dependencies retain
their respective licenses. An environment lock or a successful installation does
not by itself establish redistribution compliance. In particular, binary GPL
dependencies and NVIDIA/CUDA packages require their own redistribution review.

GWOSC strain, LVK PE and lensing-search tables, GW-LMC catalogs and cited figures
retain their upstream terms and citations. Their availability for download is not
a grant to relicense them as LAWS-Lens data. Use the precise input acquisition
manifest and upstream records.

The third-party audit is recorded separately in `release/`; this notice does not
assert that every archived binary, template or data excerpt is cleared for public
redistribution. Unresolved items block the affected Zenodo publication.

See `release/THIRD_PARTY_REVIEW_20260925_CN.md` and the wheel SBOM for the dated
findings. A hash match in an upstream download index is provenance evidence,
not certification of binary redistribution or source-offer compliance.
