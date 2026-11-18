#!/usr/bin/env python3
"""Check the adapter-supported eager reload of a shared-tool checkpoint."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from check_ouro import clean_json
from ouro_numerics import make_model

from tests.kit.numerics import Trace, judge_equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    case = json.loads((args.run / "case.json").read_text())["case"]
    saved = Trace.load(args.run / "traces/saved.safetensors")
    model_class = type(make_model(case["model"], case["loops"], case["size"]))
    model = model_class.from_pretrained(args.run / "traces/checkpoint/model", attn_implementation="eager")
    expected = {k: v for k, v in saved.items() if k.startswith("param/")}
    actual = {f"param/{n}": p.detach().cpu().to(expected[f"param/{n}"].dtype) for n, p in model.named_parameters()}
    report = judge_equal(actual, expected)
    result = dict(
        status="passed" if report.passed else "failed",
        load_options={"attn_implementation": "eager"},
        entries=[asdict(e) for e in report.entries],
        original_report_unchanged=str(args.run / "report.json"),
    )
    (args.run / "reload-eager.json").write_text(json.dumps(clean_json(result), indent=2))
    print(report.format(limit=8))
    print("EAGER_RELOAD", result["status"], args.run)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
