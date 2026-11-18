"""Summarize persisted 2D4 JSON verdicts without rerunning model computation."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()
for path in sorted(Path(args.root).rglob("*.json")):
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError):
        continue
    if "passed" in value or "status" in value and ("entries" in value or "failures" in value):
        failures = [entry for entry in value.get("entries", []) if not entry.get("passed", True)]
        print(path, value.get("passed", value.get("status")), "tensors=", len(value.get("entries", [])))
        for failure in failures[:8]:
            print(json.dumps(failure))
