"""Batches and losses that model lines can use in their ``BenchmarkSpec`` beyond the default random token ids.

- ``PackedText``: real text from a dataset on the Hub, tokenized and packed into the fixed shape. A MoE layer routes
  each token by its content, so random token ids spread tokens over the experts unlike real text.
- ``LatentVideo`` with ``flow_matching_loss`` and ``LatentTokens``: one training step of a video diffusion transformer
  on latents, as a model line trains it with the VAE and text encoder run beforehand: noised latents, a timestep and
  text embeddings in, the flow-matching velocity out.

All are dataclasses of plain values, so a spec using them can be sent to the measuring processes.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class PackedText:
    """Text of ``dataset`` (a parquet file of a Hub dataset repository, pinned to a revision), tokenized with the
    tokenizer of ``tokenizer`` and packed: documents are joined with the end-of-sequence token and cut into
    ``micro_batch × seq_len`` tokens, the inputs being their own labels. Each rank starts at a different document
    (from its generator), so ranks train different text."""

    tokenizer: str
    tokenizer_revision: str
    dataset: str
    dataset_revision: str
    file: str
    column: str = "text"
    # documents a rank may start from
    starts: int = 1000

    def __call__(
        self, model: nn.Module, micro_batch: int, seq_len: int, generator: torch.Generator
    ) -> Dict[str, Tensor]:
        needed = micro_batch * seq_len
        start = int(torch.randint(self.starts, (1,), generator=generator))
        ids = torch.tensor(_packed(self, start, needed), dtype=torch.long).view(micro_batch, seq_len)
        return dict(input_ids=ids, labels=ids.clone())


@lru_cache(maxsize=8)
def _packed(text: PackedText, start: int, needed: int) -> List[int]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(text.tokenizer, revision=text.tokenizer_revision)
    path = hf_hub_download(text.dataset, text.file, repo_type="dataset", revision=text.dataset_revision)
    ids: List[int] = []
    skipped = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=[text.column], batch_size=256):
        for document in batch.column(0).to_pylist():
            if skipped < start:
                skipped += 1
                continue
            ids += tokenizer(document)["input_ids"] + [tokenizer.eos_token_id]
            if len(ids) >= needed:
                return ids[:needed]
    raise ValueError(f"{text.file} has fewer than {needed} tokens after document {start}")


@dataclass(frozen=True)
class LatentVideo:
    """A batch of video latents of fixed shape for a diffusion transformer trained with flow matching: clean latents
    ``x0`` and noise, mixed at a random time ``t`` in [0, 1] as ``(1 − t) · x0 + t · noise``, the target velocity
    ``noise − x0``, the timestep ``1000 · t`` and random text embeddings. The latents are random: a dense transformer
    computes the same whatever their values.

    The shape is the latent one, after the VAE: ``frames × height × width`` latents of ``channels`` channels, which the
    transformer cuts into patches of ``patch``; ``tokens`` is the sequence length it attends over. The sequence length
    of the setup must equal it: a video's sequence is set by its frames and resolution."""

    channels: int
    frames: int
    height: int
    width: int
    patch: Tuple[int, int, int]
    text_len: int
    text_dim: int

    @property
    def tokens(self) -> int:
        return (self.frames // self.patch[0]) * (self.height // self.patch[1]) * (self.width // self.patch[2])

    def __call__(self, model: nn.Module, micro_batch: int, seq_len: int, generator: torch.Generator) -> Dict[str, Any]:
        if seq_len != self.tokens:
            raise ValueError(
                f"a video of {self.frames}×{self.height}×{self.width} latents in patches of {self.patch} is "
                f"{self.tokens} tokens, not {seq_len}: its sequence length is set by its frames and resolution"
            )
        shape = (micro_batch, self.channels, self.frames, self.height, self.width)
        clean, noise = torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)
        t = torch.rand(micro_batch, generator=generator)
        mix = t.view(-1, 1, 1, 1, 1)
        return dict(
            hidden_states=(1 - mix) * clean + mix * noise,
            timestep=t * 1000,
            encoder_hidden_states=torch.randn(micro_batch, self.text_len, self.text_dim, generator=generator),
            target=noise - clean,
        )


def flow_matching_loss(model: nn.Module, batch: Dict[str, Any]) -> Tensor:
    """Mean squared error between the predicted and the target velocity of a ``LatentVideo`` batch."""
    inputs = {k: v for k, v in batch.items() if k != "target"}
    prediction = model(**inputs, return_dict=False)[0]
    return F.mse_loss(prediction.float(), batch["target"].float())


@dataclass(frozen=True)
class LatentTokens:
    """The tokens a ``LatentVideo`` batch counts for throughput: latent patches, over the micro-batch."""

    patch: Tuple[int, int, int]

    def __call__(self, batch: Dict[str, Any]) -> int:
        samples, _, frames, height, width = batch["hidden_states"].shape
        return samples * (frames // self.patch[0]) * (height // self.patch[1]) * (width // self.patch[2])
