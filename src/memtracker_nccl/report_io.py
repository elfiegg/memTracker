"""Read JSON evidence, optionally gzip compressed for large owner/trace reports."""
import gzip
import json
from pathlib import Path


def read_report(path: Path):
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def write_report(path: Path, report):
    raw = (json.dumps(report, indent=2) + "\n").encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
