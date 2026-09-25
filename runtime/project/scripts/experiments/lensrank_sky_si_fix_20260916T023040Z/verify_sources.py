"""Check the corrected conversion on every distinct frozen source."""
import argparse
import json
from pathlib import Path

import bilby
import lal
import numpy as np
import pandas as pd

from test_spin_units import function
import repair_experiment as r


def run(root):
    c = r.verify(root)
    convert = function()
    seen, rows = set(), []
    for group in c["groups"]:
        for item in group["records"]:
            identity = (group["deployment"], item["record"]["source_uid"])
            if identity in seen:
                continue
            seen.add(identity)
            source = pd.Series(item["source"])
            args = [float(source[k]) for k in ["theta_jn", "phijl", "tilt1", "tilt2", "phi12", "a1", "a2"]]
            reference = bilby.gw.conversion.bilby_to_lalsimulation_spins(
                *args, float(source.m1_det) * lal.MSUN_SI,
                float(source.m2_det) * lal.MSUN_SI, 20., float(source.phase))
            angle, spins = convert(source)
            actual = np.asarray([angle, *spins])
            np.testing.assert_array_equal(actual, reference)
            if not np.isfinite(actual).all() or not 0 <= angle <= np.pi:
                raise RuntimeError("Invalid transformed source")
            error = max(abs(np.linalg.norm(spins[:3]) - source.a1),
                        abs(np.linalg.norm(spins[3:]) - source.a2))
            if error > 1e-12:
                raise RuntimeError("Spin magnitude not preserved")
            rows.append({"deployment": identity[0], "source_uid": identity[1], "iota": angle,
                         "max_abs_reference_difference": float(np.max(abs(actual - reference))),
                         "max_spin_norm_error": float(error)})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "tables/all_source_SI_conversion_checks.csv", index=False, encoding="utf-8-sig")
    status = {"state": "PASS", "distinct_run_source_ids": len(frame),
              "per_run": frame.groupby("deployment").size().to_dict(),
              "max_abs_reference_difference": float(frame.max_abs_reference_difference.max()),
              "max_spin_norm_error": float(frame.max_spin_norm_error.max()),
              "independent_lens_population_validation": False}
    r.write(root / "contracts/ALL_SOURCE_CONVERSION_CHECK.json", status)
    print(json.dumps(status))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root)
