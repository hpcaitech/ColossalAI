import math
from typing import Callable

from colossalai.utils import get_current_device
from torch import dtype, nn

from ... import init as init
from ..parallel_1d import *
from ..parallel_2d import *
from ..parallel_2p5d import *
from ..parallel_3d import *
from ..utils import get_tensor_parallel_mode
from ..vanilla import *

_parallel_embedding = {
    '2d': Embedding2D,
    '2.5d': Embedding2p5D,
    '3d': Embedding3D,
}

_vocab_parallel_embedding = {
    '1d': VocabParallelEmbedding1D,
    '2d': VocabParallelEmbedding2D,
    '2.5d': VocabParallelEmbedding2p5D,
    '3d': VocabParallelEmbedding3D
}

_parallel_patchembedding = {
    None: VanillaPatchEmbedding,
    '1d': VanillaPatchEmbedding,
    '2d': PatchEmbedding2D,
    '2.5d': PatchEmbedding2p5D,
    '3d': PatchEmbedding3D
}


class Embedding(nn.Module):
    """
    Embedding for colossalai

    :param num_embeddings: number of embeddings
    :type num_embeddings: int
    :param embedding_dim: dimension of embedding
    :type embedding_dim: int
    :param padding_idx: index of padding, defaults to None
    :type padding_idx: int, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param weight_initializer: The intializer of weight, defaults to normal initializer
    :type weight_initializer: typing.Callable, optional
    :param args: Args used in F.embedding
    :param kwargs: Kwargs used in F.embedding
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 vocab_parallel_limit: int = 2048,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel is None or (tensor_parallel == '1d' and num_embeddings <= vocab_parallel_limit):
            self.embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, *args,
                                      **kwargs).to(dtype).to(get_current_device())
            weight_initializer(self.embed.weight, fan_in=num_embeddings, fan_out=embedding_dim)
        elif num_embeddings <= vocab_parallel_limit:
            self.embed = _parallel_embedding[tensor_parallel](
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
                dtype=dtype,
                weight_initializer=weight_initializer,
                *args,
                **kwargs,
            )
        else:
            self.embed = _vocab_parallel_embedding[tensor_parallel](
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
                dtype=dtype,
                weight_initializer=weight_initializer,
                *args,
                **kwargs,
            )

    @property
    def weight(self):
        return self.embed.weight

    def forward(self, *args):
        return self.embed(*args)


class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding

    :param img_size: image size
    :type img_size: int
    :param patch_size: patch size
    :type patch_size: int
    :param in_chans: number of channels of input image
    :type in_chans: int
    :param embed_size: size of embedding
    :type embed_size: int
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param flatten: whether to flatten output tensor, defaults to True
    :type flatten: bool, optional
    :param weight_initializer: The intializer of weight, defaults to kaiming uniform initializer
    :type weight_initializer: typing.Callable, optional
    :param bias_initializer: The intializer of bias, defaults to xavier uniform initializer
    :type bias_initializer: typing.Callable, optional
    :param position_embed_initializer: The intializer of position embedding, defaults to zero
    :type position_embed_initializer: typing.Callable, optional
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
        position_embed_initializer: Callable = init.zeros_()
    ) -> None:
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        self.embed = _parallel_patchembedding[tensor_parallel](
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

    @property
    def weight(self):
        return self.embed.weight

    @property
    def bias(self):
        return self.embed.bias

    @property
    def pos_embed(self):
        return self.embed.pos_embed

    @property
    def cls_token(self):
        return self.embed.cls_token

    def forward(self, *args):
        return self.embed(*args)
