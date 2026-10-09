# Real-event sample used by LAWS-Lens (GWLR-UC-01)

`real_event_manifest.csv` lists all 222 event records used for the current real-catalog rankings: 62 O3, 74 O4a and 86 O4b events. All unordered pairs within each run are scored. This is an included-event manifest, not a claim of complete coverage of each public catalog.

## Fields

- `run`: observing-run sample (O3, O4a or O4b).
- `event_name`: public event identifier and join key within the run.
- `gps_time`: event time in GPS seconds, copied from the analysis input manifest.
- `detectors`: detector metadata recorded in that manifest. This is not a reconstruction of the detectors used for every posterior or sky product. H1/L1 are the waveform inputs.
- `network_snr`: catalog SNR metadata, not a new selection threshold.
- `pe_group`: selected group in the public posterior file.
- `pe_file`, `map_file`: basenames of input products. No local server paths are required to read this table.
- `map_format`: recorded sky-product format.

## Provenance and selection

The CSV is a field-preserving export of `contracts/REAL_INPUT_MANIFEST.json` from the completed GWLR-UC-01 archive. Source SHA-256:

`ff15bb72fef0b07cc1291478a4bc1fcdb6bc6e58f2012095102532f854451a80`

O3 starts from the 70-event LVK search sample. Nominal H1/L1 coverage leaves 63 events; GW190725_174728 fails the finite H1 waveform-window requirement, leaving 62. There is no additional SNR threshold.

O4a uses the existing 74-event GWTC-4.1 H1/L1 input sample, rechecked for waveform preprocessing by the current analysis. Its original source manifest is the archived `real_gwtc34_lensing_search_20260629_full_o4/data/event_manifest.csv`. This compact export does not contain a complete exclusion audit of every event in the full release. Such an audit is needed before claiming this is the exhaustive eligible sample.

O4b uses the intersection of 104 Stable Release 9 PE events with events listing both H1 and L1 in GWOSC, giving 86 events. The original scope is recorded in `gwtc5_o4b_frozen_manifest.csv`. An additional historical DATA-segment metadata audit returned 85 passes; this stricter flag did not define the 86-event sample. It must not be conflated with nominal detector membership or successful subsequent waveform processing.

Posterior parameter agreement and published lensing-candidate scores do not filter these pair lists. The selected sky maps supply ranking inputs; other PE parameters are compared after ranking.

## Access and reuse status

This CSV accompanies the manuscript source. It is not a public repository deposit and has no newly assigned DOI. It does not contain strain, full posterior samples, sky arrays, or their access URLs. The complete raw-input access manifest, versioned public deposit and reuse licence remain author release tasks. Existing citations to GWTC data releases identify the reused public products; no new licence for third-party data is asserted here.
