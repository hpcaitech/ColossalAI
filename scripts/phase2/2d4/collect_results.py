"""Collect completed 2D4 JUnit, numerical verdicts and full-model smoke records."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def collect(root):
    result = {"directory": str(root)}
    junit = root / "junit.xml"
    if junit.exists():
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        result["pytest"] = {
            key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")
        }
    records = []
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(value, dict) and "passed" in value and "entries" in value:
            entries = value["entries"]
            records.append(
                dict(
                    file=str(path.relative_to(root)),
                    passed=value["passed"],
                    checks=len(entries),
                    failures=[entry["name"] for entry in entries if not entry["passed"]],
                )
            )
        elif isinstance(value, dict) and path.name in ("ouro14.json", "ouro26.json") and "status" in value:
            result.setdefault("full_models", []).append(value)
    if records:
        result["numerical_reports"] = records
    result["complete"] = "pytest" in result or bool(result.get("full_models"))
    result["passed"] = (
        result["complete"]
        and not any(result.get("pytest", {}).get(k, 0) for k in ("failures", "errors", "skipped"))
        and all(record["passed"] for record in records)
        and all(model["status"] == "passed" for model in result.get("full_models", []))
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    reports = [collect(root) for root in args.directories]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        numerics = report.get("numerical_reports", [])
        print(
            json.dumps(
                dict(
                    directory=report["directory"],
                    complete=report["complete"],
                    passed=report["passed"],
                    pytest=report.get("pytest"),
                    numerical_reports=len(numerics),
                    numerical_failures=sum(not n["passed"] for n in numerics),
                    full_models=report.get("full_models", []),
                )
            )
        )

    if args.require_pass and not all(report["passed"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
