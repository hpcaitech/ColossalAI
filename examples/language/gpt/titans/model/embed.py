import torch
import torch.nn.init as init
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from colossalai.accelerator import get_accelerator
from colossalai.legacy.context import ParallelMode, seed
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.nn.layer.base_layer import ParallelLayer
from colossalai.legacy.nn.layer.parallel_1d._utils import gather_forward_split_backward, reduce_grad, reduce_input
from colossalai.legacy.nn.layer.parallel_1d.layers import Linear1D_Row
from colossalai.legacy.nn.layer.utils import divide
from colossalai.legacy.registry import LAYERS, LOSSES


class VocabParallelEmbedding(torch.nn.Module):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self, hidden_size, vocab_size, max_sequence_length, embedding_dropout_prob, num_tokentypes=0, dtype=torch.float
    ):
        super(VocabParallelEmbedding, self).__init__()

        self.hidden_size = hidden_size
        self.num_tokentypes = num_tokentypes

        # Word embeddings (parallel).
        self.word_embeddings = VocabParallelEmbedding1D(vocab_size, self.hidden_size, dtype=dtype)
        self._word_embeddings_key = "word_embeddings"

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size, dtype=dtype)
        self._position_embeddings_key = "position_embeddings"
        # Initialize the position embeddings.
        # self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size, dtype=dtype)
            # Initialize the token-type embeddings.
            # self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        if torch.distributed.get_rank() == 0:
            print("adding embedding for {} tokentypes".format(num_tokentypes), flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        # self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        # Embeddings.
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        words_embeddings = self.word_embeddings(input_ids)

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        if position_ids is None:
            position_ids = torch.arange(
                0, input_shape[-1] + 0, dtype=torch.long, device=get_accelerator().get_current_device()
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings

        # Dropout.
        with seed(ParallelMode.TENSOR):
            embeddings = self.embedding_dropout(embeddings)
        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix="", keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] = self.position_embeddings.state_dict(destination, prefix, keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = self.tokentype_embeddings.state_dict(
                destination, prefix, keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "word_embeddings" in key:
                    state_dict_[key.split("word_embeddings.")[1]] = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "position_embeddings" in key:
                    state_dict_[key.split("position_embeddings.")[1]] = state_dict[key]
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if "tokentype_embeddings" in key:
                        state_dict_[key.split("tokentype_embeddings.")[1]] = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_, strict=strict)
            else:
                print(
                    "***WARNING*** expected tokentype embeddings in the " "checkpoint but could not find it", flush=True
                )


class VocabParallelEmbedding1D(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, dtype=None, init_method=None):
        super(VocabParallelEmbedding1D, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the details for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = gpc.tensor_parallel_size
        # Divide the weight matrix along the vocabulary dimension.
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, gpc.get_local_rank(ParallelMode.PARALLEL_1D), self.tensor_model_parallel_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # Allocate weights and initialize.
        factory_kwargs = {"device": get_accelerator().get_current_device(), "dtype": dtype}
        self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim, **factory_kwargs))
        init.uniform_(self.weight, -1, 1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = output = reduce_input(output_parallel, ParallelMode.PARALLEL_1D)
        return output


@LOSSES.register_module
class vocab_parallel_cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vocab_parallel_logits, target):
        """Helper function for the cross entropy."""
        vocab_parallel_logits = vocab_parallel_logits[..., :-1, :].contiguous()
        target = target[..., 1:].contiguous()
        return _VocabParallelCrossEntropy.apply(
            vocab_parallel_logits.view(-1, vocab_parallel_logits.size(-1)), target.view(-1)
        )


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=gpc.get_group(ParallelMode.PARALLEL_1D)
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
        world_size = gpc.tensor_parallel_size
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits, op=torch.distributed.ReduceOp.SUM, group=gpc.get_group(ParallelMode.PARALLEL_1D)
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=gpc.get_group(ParallelMode.PARALLEL_1D)
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits
        loss = loss.mean()
        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)


class VocabParallelGPTLMHead1D(ParallelLayer):
    """
    Language model head that shares the same parameters with the embedding matrix.
    """

    def __init__(self, embed=None, vocab_size=None, dtype=None, embed_dim=None):
        super().__init__()
        if embed is not None:
            self.head = embed
        else:
            self.head = VocabParallelEmbedding1D(vocab_size, embed_dim, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = reduce_grad(x, ParallelMode.PARALLEL_1D)
        x = F.linear(x, self.head.weight)
        return x


###################################


class HiddenParallelEmbedding(torch.nn.Module):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        dtype=torch.float,
        padding_idx: int = 0,
        num_tokentypes=0,
    ):
        super(HiddenParallelEmbedding, self).__init__()

        self.hidden_size = hidden_size
        self.num_tokentypes = num_tokentypes

        # Word embeddings (parallel).
        self.word_embeddings = HiddenParallelEmbedding1D(vocab_size, hidden_size, dtype, padding_idx)
        self._word_embeddings_key = "word_embeddings"

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)
        self._position_embeddings_key = "position_embeddings"
        # Initialize the position embeddings.
        # self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
            # Initialize the token-type embeddings.
            # self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        if torch.distributed.get_rank() == 0:
            print("adding embedding for {} tokentypes".format(num_tokentypes), flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        # self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        words_embeddings = self.word_embeddings(input_ids)

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        if position_ids is None:
            position_ids = torch.arange(
                0, input_shape[-1] + 0, dtype=torch.long, device=get_accelerator().get_current_device()
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings

        # Dropout.
        with seed(ParallelMode.TENSOR):
            embeddings = self.embedding_dropout(embeddings)
        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix="", keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] = self.position_embeddings.state_dict(destination, prefix, keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = self.tokentype_embeddings.state_dict(
                destination, prefix, keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "word_embeddings" in key:
                    state_dict_[key.split("word_embeddings.")[1]] = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "position_embeddings" in key:
                    state_dict_[key.split("position_embeddings.")[1]] = state_dict[key]
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if "tokentype_embeddings" in key:
                        state_dict_[key.split("tokentype_embeddings.")[1]] = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_, strict=strict)
            else:
                print(
                    "***WARNING*** expected tokentype embeddings in the " "checkpoint but could not find it", flush=True
                )


class HiddenParallelEmbedding1D(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, dtype=torch.float, padding_idx: int = None, init_method=None):
        super(HiddenParallelEmbedding1D, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        embed_dim_per_partition = divide(embedding_dim, gpc.tensor_parallel_size)
        # Set the details for compatibility.
        self.padding_idx = padding_idx
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        # Allocate weights and initialize.
        factory_kwargs = {"device": get_accelerator().get_current_device(), "dtype": dtype}
        self.weight = Parameter(torch.empty(num_embeddings, embed_dim_per_partition, **factory_kwargs))
        init.uniform_(self.weight, -1, 1)

    def forward(self, input_):
        # Get the embeddings.
        output_parallel = F.embedding(
            input_, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
        )

        # Reduce across all the model parallel GPUs.
        output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)
        return output


@LAYERS.register_module
class HiddenParallelGPTLMHead1D(ParallelLayer):
    """
    Language model head that shares the same parameters with the embedding matrix.
    """

    def __init__(
        self,
        embed=None,
        embed_dim=None,
        vocab_size=None,
        dtype=None,
    ):
        super().__init__()
        if embed is not None:
            self.head = embed
            self.synced_embed = True
        else:
            # self.embedding = HiddenParallelEmbedding1D(vocab_size, hidden_size, dtype, padding_idx)
            # (hidden_size/q, vocab_size)
            self.synced_embed = False
            self.head = Linear1D_Row(
                in_features=embed_dim, out_features=vocab_size, bias=False, dtype=dtype, parallel_input=False
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.synced_embed:
            x = F.linear(x, self.head.weight)
        else:
            x = self.head(x)

        return x
