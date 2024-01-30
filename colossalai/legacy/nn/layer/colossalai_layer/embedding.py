import math
from typing import Callable

from torch import dtype, nn

from colossalai.accelerator import get_accelerator
from colossalai.nn import init

from ..parallel_1d import Embedding1D, PatchEmbedding1D, VocabParallelEmbedding1D
from ..parallel_2d import Embedding2D, PatchEmbedding2D, VocabParallelEmbedding2D
from ..parallel_2p5d import Embedding2p5D, PatchEmbedding2p5D, VocabParallelEmbedding2p5D
from ..parallel_3d import Embedding3D, PatchEmbedding3D, VocabParallelEmbedding3D
from ..utils import get_tensor_parallel_mode
from ..vanilla import VanillaPatchEmbedding
from ._utils import ColossalaiModule

_parallel_embedding = {
    "1d": Embedding1D,
    "2d": Embedding2D,
    "2.5d": Embedding2p5D,
    "3d": Embedding3D,
}

_vocab_parallel_embedding = {
    "1d": VocabParallelEmbedding1D,
    "2d": VocabParallelEmbedding2D,
    "2.5d": VocabParallelEmbedding2p5D,
    "3d": VocabParallelEmbedding3D,
}

_parallel_patchembedding = {
    None: VanillaPatchEmbedding,
    "1d": PatchEmbedding1D,
    "2d": PatchEmbedding2D,
    "2.5d": PatchEmbedding2p5D,
    "3d": PatchEmbedding3D,
}


class Embedding(ColossalaiModule):
    r"""Embedding for colossalai.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_dim (int): dimension of embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx do not contribute to the gradient;
            therefore, the embedding vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”, defaults to None.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            he initializer of weight, defaults to normal initializer.

    The ``args`` and ``kwargs`` used in :class:`torch.nn.functional.embedding` should contain:
    ::

        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is
                    renormalized to have norm max_norm. Note: this will modify weight in-place.
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse
                    of frequency of the words in the mini-batch. Default False.
        sparse (bool, optional): If True, gradient w.r.t. weight will be a sparse tensor. Default False.

    More details about ``args`` and ``kwargs`` could be found in
    `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html#torch.nn.functional.embedding>`_.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        dtype: dtype = None,
        weight_initializer: Callable = init.normal_(),
        vocab_parallel_limit: int = 2048,
        *args,
        **kwargs,
    ) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel is None:
            embed = (
                nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, *args, **kwargs)
                .to(dtype)
                .to(get_accelerator().get_current_device())
            )
            weight_initializer(embed.weight, fan_in=num_embeddings, fan_out=embedding_dim)
        elif num_embeddings <= vocab_parallel_limit:
            embed = _parallel_embedding[tensor_parallel](
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
                dtype=dtype,
                weight_initializer=weight_initializer,
                *args,
                **kwargs,
            )
        else:
            embed = _vocab_parallel_embedding[tensor_parallel](
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
                dtype=dtype,
                weight_initializer=weight_initializer,
                *args,
                **kwargs,
            )
        super().__init__(embed)


class PatchEmbedding(ColossalaiModule):
    """2D Image to Patch Embedding.

    Args:
        img_size (int): image size.
        patch_size (int): patch size.
        in_chans (int): number of channels of input image.
        embed_size (int): size of embedding.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        flatten (bool, optional): whether to flatten output tensor, defaults to True.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.
        position_embed_initializer (:class:`typing.Callable`, optional):
            The initializer of position embedding, defaults to zeros initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_size: int,
        dtype: dtype = None,
        flatten: bool = True,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
        position_embed_initializer: Callable = init.zeros_(),
    ) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        embed = _parallel_patchembedding[tensor_parallel](
            img_size,
            patch_size,
            in_chans,
            embed_size,
            dtype=dtype,
            flatten=flatten,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            position_embed_initializer=position_embed_initializer,
        )
        super().__init__(embed)
