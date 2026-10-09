# Code entry points

| Calculation | Implementation |
| --- | --- |
| Source waveform, lensing and detector response | runtime/training/scripts/source_generator.py |
| Signals in measured noise and waveform inputs | runtime/training/scripts/c_data.py |
| Simulation/training entry | runtime/training/scripts/unified_ab.py, which loads c_data |
| Matched-filter recovery and BAYESTAR | runtime/training/scripts/c_trigger.py |
| Calibrated waveform, time, sky and combined scores | runtime/completion/scripts/uab_scoring.py |
| Injection retrieval and ranking | runtime/completion/scripts/uab_evaluate.py |
| Observed events and PE comparisons | runtime/completion/scripts/uab_real.py |
| Representative injection regeneration | scripts/verify_injections.py |
| Release checks | scripts/reproduce.py; extension/scripts/reproduce_extension.py |

Use the portable wrappers for reproduction. Original directory names remain
in the dependency tree because the analysis imports shared modules from it.

## Development code

`runtime/project/scripts/server_experiments/p2c_noise_and_null.py` contains an
unfinished noise adapter, outside the paper reproduction entry points. Its
NotImplementedError marks missing input integration, not a reported result.

`runtime/project/matchgw/catalog.py` initializes recall and F1 before
`catalog_system_report` counts recovered systems and `finalize_catalog_metrics`
computes final values. Those temporary fields are not the paper's reported
retrieval metrics, which are evaluated through `uab_evaluate.py`.

Static source checks parse Python and inspect entry points. Numerical validation
requires the data, dependencies and workflow execution described in REPRODUCTION.md.
