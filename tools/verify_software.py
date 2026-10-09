"""Check release source files and documented editorial updates."""
import argparse
import hashlib
import json
from pathlib import Path


def matches(data, row):
    """Accept Git text line-ending conversion, but no other content change."""
    variants = [data]
    if b'\x00' not in data:
        lf = data.replace(b'\r\n', b'\n')
        variants.extend([lf, lf.replace(b'\n', b'\r\n')])
    return any(len(v) == row['bytes'] and hashlib.sha256(v).hexdigest() == row['sha256'] for v in variants)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', action='store_true', help='Check the original archived source without editorial updates')
    args = parser.parse_args()
    manifest = json.loads((root / "release/SOFTWARE_SOURCE_MANIFEST.json").read_text())
    update_path = root / 'release/SOURCE_EDITORIAL_UPDATES.json'
    updates = {} if args.original or not update_path.exists() else {
        row['path']: row for row in json.loads(update_path.read_text())['files']}
    failures = []
    for row in manifest["files"]:
        p = root / row["path"]
        if row['path'] in updates:
            update = updates[row['path']]
            if update['original_sha256'] != row['sha256']:
                failures.append({'path':row['path'], 'error':'editorial update has a different baseline'})
                continue
            row = update
        if not p.is_file():
            failures.append({"path": row["path"], "error": "missing"})
        elif not matches(p.read_bytes(), row):
            failures.append({"path": row["path"], "error": "checksum mismatch"})
    print(json.dumps({"status": "FAIL" if failures else "PASS", "files": len(manifest["files"]),
        "documented_editorial_updates": len(updates), "text_line_endings": "LF or CRLF", "failures": failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
