# Five-size O3 runtime curve

The current `figures/fig_runtime_scaling.pdf` uses the five measurements in
`runtime_summary.csv`. Both panels show medians with full minimum-to-maximum
ranges: three initial-processing measurements and nine cached pair-scoring
measurements for each of 5, 15, 30, 46 and 62 events.

`scaling_timings.csv` contains all 75 timing records; `stage_timings.csv`
contains the preparation-stage measurements. Statistics were recomputed from
the individual records and matched the supplied summary. Inference was not
rerun during plotting. The implementation is DOMAIN04/PATH1.

Full evidence archive: `LAWS_O3_FULL_RUNTIME_CURVE_EVIDENCE.zip`

SHA-256: `b5f2612121b62316c86860ddef4ad22b8dc7af01f7d0f49ca4c76a427107e8c5`

To reproduce, safely extract that archive and run:

```text
python plot_runtime_curve.py --data-dir EXTRACTED_ARCHIVE --output-dir OUTPUT_DIRECTORY
```

The script verifies all 263 manifest entries before plotting. The manuscript
uses 74.9 s and 0.849 s in the abstract, rounded from medians 74.9328580680 s
and 0.8494315408 s. The complete measurement ranges remain in the figure.

These measurements support the runtime figure and the timing summaries in
the manuscript and Supplementary Information. Earlier measurements are retained
in the historical archives listed in the version map.
