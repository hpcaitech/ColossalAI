#!/usr/bin/env python3
"""Summarize preserved 2D4 cases without rerunning any model."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def summarize(root):
    reports = []
    for path in sorted(root.glob("run-*/report.json")):
        report = json.loads(path.read_text())
        report["report_path"] = str(path)
        reports.append(report)
    rows = []
    for report in reports:
        case = report["case"]
        entries = report.get("entries", [])
        for rank in report.get("ranks", []):
            entries.extend(dict(item, name=f"rank{rank['rank']}/{item['name']}") for item in rank["entries"])
        failures = [e for e in entries if not e["passed"]]
        ratios = [
            e["test_error"] / e["baseline_error"]
            for e in entries
            if e.get("baseline_error") and e.get("test_error") is not None
        ]
        row = dict(
            model=case["model"],
            loops=case["loops"],
            size=case["size"],
            scenario=case["scenario"],
            plugin="zero1_bf16" if case["scenario"] == "blank" else case["plugin"],
            world_size=case["world_size"],
            micro_batches=case["micro_batches"],
            steps=case["steps"],
            seed=case["seed"],
            status=report["status"],
            tensors=len(entries),
            failed_tensors=len(failures),
            max_ratio=max(ratios, default=None),
            first_failures=[dict(name=e["name"], reason=e["reason"], metrics=e.get("metrics")) for e in failures[:5]],
            error=report.get("error"),
            report=report["report_path"],
        )
        rows.append(row)
    reloads = []
    for path in sorted(root.glob("run-*/reload-eager.json")):
        data = json.loads(path.read_text())
        reloads.append(dict(report=str(path), status=data["status"], tensors=len(data["entries"])))
    return dict(
        schema_version="2d4-probes-v1",
        created=datetime.now(timezone.utc).isoformat(),
        source=reports[0]["source"] if reports else {},
        source_audits=[str(p) for p in sorted(root.glob("source-audit-*.json"))],
        eager_reload_checks=reloads,
        scope="2D4 probes of current feature working tree; tiny/small random initialization, not full L1/L2 acceptance",
        summary=dict(Counter(r["status"] for r in rows)),
        results=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/results/2d4"))
    parser.add_argument("--out", type=Path, help="Optional named export; existing files are not overwritten")
    args = parser.parse_args()
    report = summarize(args.root)
    out = args.out or args.root / ("summary-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".json")
    with out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False))
    for row in report["results"]:
        print(
            row["status"],
            row["model"],
            row["loops"],
            row["scenario"],
            row["plugin"],
            row["world_size"],
            row["micro_batches"],
            f'{row["tensors"]-row["failed_tensors"]}/{row["tensors"]}',
            row["report"],
            flush=True,
        )
        if row["error"]:
            print("ERROR", row["error"][-1300:], flush=True)
        for failure in row["first_failures"]:
            print("FAIL", json.dumps(failure), flush=True)
    print("SUMMARY", report["summary"], out)


if __name__ == "__main__":
    main()
