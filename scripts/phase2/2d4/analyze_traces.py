#!/usr/bin/env python3
"""Quantify differences in already-saved DP or resume traces, without training."""
import argparse
import json
from pathlib import Path

import torch

from tests.kit.numerics import Trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    case = json.loads((args.run / "case.json").read_text())["case"]
    if case["scenario"] == "resume":
        actual = Trace.load(args.run / "traces/resumed.safetensors")
        reference = Trace.load(args.run / "traces/uninterrupted.safetensors")
    else:
        actual = Trace.load(args.run / "traces/test.safetensors")
        reference = Trace.load(args.run / "traces/ref.safetensors")
    rows = []
    for name, value in actual.items():
        if name not in reference:
            continue
        other = reference[name]
        if torch.equal(value, other):
            continue
        left, right = value.double(), other.double()
        difference = left - right
        denominator = right.norm().item()
        rows.append(
            dict(
                name=name,
                dtype=str(value.dtype),
                max_abs=difference.abs().max().item(),
                relative_l2=difference.norm().item() / denominator if denominator else None,
                different_elements=int((value != other).sum()),
                elements=value.numel(),
            )
        )
    rows.sort(key=lambda row: row["relative_l2"] or 0, reverse=True)
    result = dict(
        case=case,
        differing_tensors=len(rows),
        differences=rows,
        maximum_absolute_error=max((r["max_abs"] for r in rows), default=0),
        maximum_relative_l2=max((r["relative_l2"] or 0 for r in rows), default=0),
    )
    (args.run / "trace-differences.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "differences"}))
    for row in rows[:5]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
