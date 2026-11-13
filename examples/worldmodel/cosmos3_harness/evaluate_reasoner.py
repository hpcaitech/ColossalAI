r"""Shared reasoner-evaluation adapter for the GPU and Ascend Cosmos 3 runs.

The official reasoner numbers use VLMEvalKit together with a deployed model endpoint. Re-implementing the prompts or
the scorer here would produce a number that cannot be compared with the model card, so this module only emits and
validates a benchmark manifest. The actual run is delegated to the repository's VLMEvalKit environment; the manifest
is the part both devices must share.

    python examples/worldmodel/cosmos3_harness/evaluate_reasoner.py --manifest manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

__all__ = ["BENCHMARKS", "build_manifest", "write_manifest", "load_responses", "score_choice"]

# The first-round subset. Every entry names the dataset exactly as VLMEvalKit does and records that the official
# score has to be filled from the pinned model card; no number is copied from a different split or prompt protocol.
BENCHMARKS = {
    "MMBench-Dev": {
        "vlmevalkit_dataset": "MMBench_DEV_EN",
        "split": "dev",
        "language": "en",
        "scorer": "MMBench exact-match circular evaluation (VLMEvalKit)",
        "official_edge_score": None,
        "note": "Fill official_edge_score from the pinned nvidia/Cosmos3-Edge revision before comparing.",
    },
    "HallusionBench": {
        "vlmevalkit_dataset": "HallusionBench",
        "split": "test",
        "language": "en",
        "scorer": "HallusionBench question-pair accuracy (VLMEvalKit)",
        "official_edge_score": None,
        "note": "The model card reports General-category benchmarks; copy the matching row from the pinned revision.",
    },
    "MMMU": {
        "vlmevalkit_dataset": "MMMU_DEV_VAL",
        "split": "validation",
        "language": "en",
        "scorer": "MMMU accuracy (VLMEvalKit)",
        "official_edge_score": None,
        "note": "Split name must be confirmed against the pinned VLMEvalKit config before reporting a comparison.",
    },
}


def build_manifest(benchmarks: Sequence[str], *, model: str, model_revision: str, decoder: Dict) -> Dict:
    """Build a device-independent evaluation manifest.

    Args:
        benchmarks (Sequence[str]): Names from :data:`BENCHMARKS`.
        model (str): Reasoner checkpoint id or local path.
        model_revision (str): Immutable checkpoint revision/commit.
        decoder (Dict): Decoding configuration (max tokens, temperature, EOS, prompt mode, chat template).

    Returns:
        Dict: Manifest with one entry per benchmark and a stable fingerprint.

    Raises:
        ValueError: If a benchmark is unknown or the decoder configuration is incomplete.
    """
    unknown = [name for name in benchmarks if name not in BENCHMARKS]
    if unknown:
        raise ValueError(f"Unknown benchmark(s) {unknown}; known names are {sorted(BENCHMARKS)}.")
    required_decoder = {"max_new_tokens", "temperature", "do_sample"}
    missing = required_decoder - set(decoder)
    if missing:
        raise ValueError(f"decoder is missing {sorted(missing)}; record it so both devices decode identically.")
    entries = {}
    for name in benchmarks:
        entry = dict(BENCHMARKS[name])
        entry["decoder"] = dict(decoder)
        entries[name] = entry
    payload = {
        "model": model,
        "model_revision": model_revision,
        "benchmarks": entries,
        "note": "The number is comparable only when the dataset, prompt, image preprocessing, decoding and scorer "
        "match this manifest. A subset of the General category is not a reproduction of General Avg.",
    }
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def write_manifest(manifest: Dict, path: str | Path) -> Path:
    """Write the manifest as JSON and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_responses(path: str | Path) -> list:
    """Load a JSONL file of per-sample responses and check the fields the scorer needs.

    Each line must carry ``sample_id``, ``benchmark`` and ``response``. Raw responses are kept verbatim; a missing
    prediction is an error rather than an empty string, because scoring it as wrong would hide a generation failure.
    """
    responses = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = [key for key in ("sample_id", "benchmark", "response") if key not in record]
            if missing:
                raise ValueError(f"{path}:{line_number} is missing {missing}.")
            if not isinstance(record["response"], str) or not record["response"].strip():
                raise ValueError(f"{path}:{line_number} has an empty response; record the failure explicitly.")
            responses.append(record)
    return responses


def score_choice(response: str, choices: Sequence[str], answer: str) -> bool:
    """Exact-match score for a multiple-choice benchmark after stripping common answer decorations.

    This is only a fallback for a smoke subset. The official run must use the dataset's VLMEvalKit scorer, because
    extraction rules differ between benchmarks.
    """
    normalized = response.strip().upper()
    for prefix in ("ANSWER:", "ANSWER IS", "THE ANSWER IS"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    if normalized and normalized[0] in "ABCDEFGH":
        return normalized[0] == str(answer).strip().upper()
    for index, choice in enumerate(choices):
        if normalized == str(choice).strip().upper():
            return "ABCDEFGH"[index] == str(answer).strip().upper()
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--model-revision", default="pending-pin")
    parser.add_argument("--benchmarks", nargs="+", default=["MMBench-Dev", "HallusionBench"])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--manifest", default="cosmos3_reasoner_manifest.json")
    args = parser.parse_args(argv)

    manifest = build_manifest(
        args.benchmarks,
        model=args.model,
        model_revision=args.model_revision,
        decoder={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": args.temperature > 0,
            "prompt_mode": "VLMEvalKit dataset prompt",
            "chat_template": "checkpoint tokenizer chat template",
            "eos": "checkpoint generation config",
        },
    )
    write_manifest(manifest, args.manifest)
    print(
        json.dumps({"manifest": args.manifest, "benchmarks": args.benchmarks, "fingerprint": manifest["fingerprint"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
