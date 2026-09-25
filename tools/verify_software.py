"""Check the frozen software subset using only the Python standard library."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "release/SOFTWARE_SOURCE_MANIFEST.json").read_text())
    failures = []
    for row in manifest["files"]:
        p = root / row["path"]
        if not p.is_file():
            failures.append({"path": row["path"], "error": "missing"})
        elif p.stat().st_size != row["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            failures.append({"path": row["path"], "error": "checksum mismatch"})
    print(json.dumps({"status": "FAIL" if failures else "PASS", "files": len(manifest["files"]), "failures": failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
