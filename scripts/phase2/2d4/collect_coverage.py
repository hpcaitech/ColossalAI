"""Combine an initial JUnit run and its targeted rerun, retaining prior failures."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def cases(path):
    result = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        key = case.attrib["classname"] + "::" + case.attrib["name"]
        failure = next((c for c in case if c.tag in ("failure", "error", "skipped")), None)
        result[key] = {"passed": failure is None, "source": str(path)}
        if failure is not None:
            result[key]["reason"] = failure.attrib.get("message", failure.text or "")[-1000:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial", type=Path)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    initial, rerun = cases(args.initial), cases(args.rerun)
    assert initial and rerun and set(rerun) <= set(initial)
    final = {**initial, **rerun}
    result = {
        "unique_cases": len(final),
        "passed": all(case["passed"] for case in final.values()),
        "initial_failures": {name: case for name, case in initial.items() if not case["passed"]},
        "rerun_cases": rerun,
        "unresolved": {name: case for name, case in final.items() if not case["passed"]},
        "note": "Coverage across runs, not a claim that the initial batch was green. See the source audit and dated report.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
