"""Independent test oracles from pinned official sources.

LLaDA formula: ML-GSAI/LLaDA GUIDELINES.md @ 9182493720ed723ef8031210d85959364e51cbe0.
dFactory's functions/loss statements are verbatim Apache-2.0 fixture excerpts.
Only the selected AST nodes are executed, not the upstream training framework.
"""

import ast
import json
from pathlib import Path

import torch

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures/diffusion/dfactory.json").read_text())


def official_function(name):
    """Compile only the named pinned function in a minimal test namespace."""
    node = ast.parse(FIXTURE[name]).body[0]
    assert isinstance(node, ast.FunctionDef) and node.name == name
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned-dfactory", "exec"), scope)
    return scope[name]


def official_masked_ce(logits, labels):
    """Execute the official same_token_labels=True loss, with one microbatch."""
    scope = {"torch": torch, "noisy_logits": logits.contiguous(), "labels": labels, "micro_batches": [None]}
    exec(compile(FIXTURE["same_token_loss"], "pinned-dfactory-loss", "exec"), scope)
    return scope["loss"]


def llada_sequence_loss(logits, labels, probability, target_mask):
    """Independent log-softmax oracle for the LLaDA per-answer estimator."""
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()
    result = []
    for row in range(labels.shape[0]):
        positions = (labels[row] != -100).nonzero().flatten()
        # Empty indexing preserves autograd and also handles a zero-target row.
        scores = logits[row, positions].log_softmax(-1)
        terms = -scores[torch.arange(len(positions), device=logits.device), labels[row, positions]]
        weighted = terms / probability[row, positions]
        result.append(weighted.sum() / max(int(target_mask[row].sum()), 1))
    return torch.stack(result)
