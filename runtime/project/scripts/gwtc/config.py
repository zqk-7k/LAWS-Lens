from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "gwtc"
DATA_DIR = REPO_ROOT / "data"
GWTC_RAW_DIR = DATA_DIR / "gwtc_raw"
GWTC3_RAW_DIR = DATA_DIR / "gwtc3_raw"
GWTC5_RAW_DIR = DATA_DIR / "gwtc5_raw"
FIGURE_DIR = REPO_ROOT / "figures_gwtc"

SOURCE_STATUS_JSON = SCRIPT_DIR / "source_status.json"
GWTC3_OBSERVABLES_CSV = DATA_DIR / "gwtc3_observables.csv"
GWTC5_OBSERVABLES_CSV = DATA_DIR / "gwtc5_observables.csv"
GWOSC_ALLEVENTS_CACHE = GWTC_RAW_DIR / "allevents.csv"

GWOSC_ALLEVENTS_CSV = "https://gwosc.org/eventapi/csv/allevents/"
ZENODO_GWTC3_RECORD = "https://zenodo.org/api/records/8177023"
ZENODO_GWTC21_RECORD = "https://zenodo.org/api/records/6513631"
ZENODO_GWTC5_CANDIDATE_RECORD = "https://zenodo.org/api/records/20276130"
ZENODO_GWTC5_PE_SEARCH = "https://zenodo.org/api/records?q=GWTC-5.0%20parameter%20estimation"

REQUEST_TIMEOUT = 90
