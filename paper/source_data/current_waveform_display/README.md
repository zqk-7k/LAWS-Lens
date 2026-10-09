# Source arrays for main-text Figure 5

These files are unchanged copies of the inputs used to draw the four current consensus-leader events. They replace no historical files. The `fig4` filename prefix comes from the figure's earlier number; the manuscript now labels this display Figure 5.

| File | Contents |
| --- | --- |
| `fig4_gwtc3_rank1_waveforms.csv` | GW190421_213856 and GW190731_140936; one H1 and one L1 standardized channel per event. |
| `fig4_gwtc4_rank1_waveforms.csv` | GW230630_125806 and GW230707_124047; one H1 and one L1 standardized channel per event. |
| `current_rank1_pe_reconstructions.csv` | Four event-specific, noise-free plus-polarization morphology curves. |
| `current_rank1_pe_reconstruction_metadata.json` | Public release records, event-file hashes, PE groups, maximum-likelihood sample indices, waveform parameters, software versions, alignment and normalization. |

All arrays contain 4,096 samples. `time_from_trigger_s` in the noisy arrays and `time_from_peak_s` in the reconstructions share a numerical grid from -1.75 s to 0.25 s, with the final endpoint excluded, at 2,048 Hz. The different column names retain the different alignment definitions: noisy inputs use the event trigger, whereas reconstructed curves place their analytic-envelope maximum at zero.

Noisy data are the saved per-detector standardized inputs, not newly processed strain. For each run, the display multiplies all four noisy channels by `0.39 / max(abs(all four channels))`. Row offsets are 3, 2, 1 and 0 for first-event H1/L1 and second-event H1/L1. A reconstructed curve is multiplied by 0.34 and plotted at its event's row offsets. The panel shows -0.80 to +0.12 s, while these files retain the full two seconds.

The reconstruction uses rippleGW IMRPhenomXPHM and the maximum finite log-likelihood sample from each recorded public PE group. Each curve is normalized independently by its analytic-envelope maximum. The same h+ morphology is repeated for H1 and L1 without detector-response projection. It is not an amplitude or detector-phase fit, a uniquely recovered signal, or an injected ground-truth waveform. These display arrays do not enter ranking or posterior-compatibility calculations.

The waveform input arrays originate from the unchanged saved `gwtc3_top03.npz` and `gwtc4_top03.npz` exports. The event identities and 4,096-point samples were checked against those exports when preparing the figure. The reconstruction metadata identify their separate public PE sources. The plotting and reconstruction implementations are retained in the separate code delivery; this directory supplies compact numerical figure inputs.
