# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import torch
import torch.nn as nn
from typing import Tuple, List, Callable, Any, Dict, Sequence, Optional


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def pts_to_distogram(pts, min_bin=2.3125, max_bin=21.6875, no_bins=64):
    boundaries = torch.linspace(
        min_bin, max_bin, no_bins - 1, device=pts.device
    )
    dists = torch.sqrt(
        torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1)
    )
    return torch.bucketize(dists, boundaries)


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)

def _fetch_dims(tree):
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError("Not supported")

    return shapes


@torch.jit.ignore
def _flat_idx_to_idx(
    flat_idx: int,
    dims: Tuple[int],
) -> Tuple[int]:
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d

    return tuple(reversed(idx))


@torch.jit.ignore
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: int,
    start_edges: Optional[Sequence[bool]] = None,
    end_edges: Optional[Sequence[bool]] = None,
) -> Sequence[Tuple[int]]:
    """ 
        Produces an ordered sequence of tensor slices that, when used in
        sequence on a tensor with shape dims, yields tensors that contain every
        leaf in the contiguous range [start, end]. Care is taken to yield a 
        short sequence of slices, and perhaps even the shortest possible (I'm 
        pretty sure it's the latter).
         
        end is INCLUSIVE. 
    """
    # start_edges and end_edges both indicate whether, starting from any given
    # dimension, the start/end index is at the top/bottom edge of the
    # corresponding tensor, modeled as a tree
    def reduce_edge_list(l):
        tally = 1
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] *= tally
            tally = l[reversed_idx]

    if(start_edges is None):
        start_edges = [s == 0 for s in start]
        reduce_edge_list(start_edges)
    if(end_edges is None):
        end_edges = [e == (d - 1) for e,d in zip(end, dims)]
        reduce_edge_list(end_edges)        

    # Base cases. Either start/end are empty and we're done, or the final,
    # one-dimensional tensor can be simply sliced
    if(len(start) == 0):
        return [tuple()]
    elif(len(start) == 1):
        return [(slice(start[0], end[0] + 1),)]

    slices = []
    path = []
 
    # Dimensions common to start and end can be selected directly
    for s,e in zip(start, end):
        if(s == e):
            path.append(slice(s, s + 1))
        else:
            break

    path = tuple(path)
    divergence_idx = len(path)

    # start == end, and we're done
    if(divergence_idx == len(dims)):
        return [tuple(path)]

    def upper():
        sdi = start[divergence_idx]
        return [
            path + (slice(sdi, sdi + 1),) + s for s in 
            _get_minimal_slice_set(
                start[divergence_idx + 1:],
                [d - 1 for d in dims[divergence_idx + 1:]],
                dims[divergence_idx + 1:],
                start_edges=start_edges[divergence_idx + 1:],
                end_edges=[1 for _ in end_edges[divergence_idx + 1:]]
            )
        ]

    def lower():
        edi = end[divergence_idx]
        return [
            path + (slice(edi, edi + 1),) + s for s in 
            _get_minimal_slice_set(
                [0 for _ in start[divergence_idx + 1:]],
                end[divergence_idx + 1:],
                dims[divergence_idx + 1:],
                start_edges=[1 for _ in start_edges[divergence_idx + 1:]],
                end_edges=end_edges[divergence_idx + 1:],
            )
        ]

    # If both start and end are at the edges of the subtree rooted at
    # divergence_idx, we can just select the whole subtree at once
    if(start_edges[divergence_idx] and end_edges[divergence_idx]):
        slices.append(
            path + (slice(start[divergence_idx], end[divergence_idx] + 1),)
        )
    # If just start is at the edge, we can grab almost all of the subtree, 
    # treating only the ragged bottom edge as an edge case
    elif(start_edges[divergence_idx]):
        slices.append(
            path + (slice(start[divergence_idx], end[divergence_idx]),)
        )
        slices.extend(lower())
    # Analogous to the previous case, but the top is ragged this time
    elif(end_edges[divergence_idx]):
        slices.extend(upper())
        slices.append(
            path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),)
        )
    # If both sides of the range are ragged, we need to handle both sides
    # separately. If there's contiguous meat in between them, we can index it
    # in one big chunk
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if(middle_ground > 1):
            slices.append(
                path + (slice(start[divergence_idx] + 1, end[divergence_idx]),)
            )
        slices.extend(lower())

    return [tuple(s) for s in slices]


@torch.jit.ignore
def _chunk_slice(
    t: torch.Tensor,
    flat_start: int,
    flat_end: int,
    no_batch_dims: int,
) -> torch.Tensor:
    """
        Equivalent to
        
            t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

        but without the need for the initial reshape call, which can be 
        memory-intensive in certain situations. The only reshape operations
        in this function are performed on sub-tensors that scale with
        (flat_end - flat_start), the chunk size.
    """

    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    # _get_minimal_slice_set is inclusive
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))

    # Get an ordered list of slices to perform
    slices = _get_minimal_slice_set(
        start_idx,
        end_idx,
        batch_dims,
    )

    sliced_tensors = [t[s] for s in slices]

    return torch.cat(
        [s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors]
    )


def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False, 
) -> Any:
    """
    Implements the "chunking" procedure described in section 1.11.8.

    Layer outputs and inputs are assumed to be simple "pytrees,"
    consisting only of (arbitrarily nested) lists, tuples, and dicts with
    torch.Tensor leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (non-nested) dictionary of keyworded inputs. All leaves must
            be tensors and must share the same batch dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch
            dimensions are specified, a "sub-batch" is defined as a single
            indexing of all batch dimensions simultaneously (s.t. the
            number of sub-batches is the product of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can
            be considered batch dimensions.
        low_mem:
            Avoids flattening potentially large input tensors. Unnecessary
            in most cases, and is ever so slightly slower than the default
            setting.
    Returns:
        The reassembled output of the layer on the inputs.
    """
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    def _prep_inputs(t):
        # TODO: make this more memory efficient. This sucks
        if(not low_mem):
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t

    prepped_inputs = tensor_tree_map(_prep_inputs, inputs)

    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d

    no_chunks = flat_batch_dim // chunk_size + (
        flat_batch_dim % chunk_size != 0
    )

    i = 0
    out = None
    for _ in range(no_chunks):
        # Chunk the input
        if(not low_mem):
            select_chunk = (
                lambda t: t[i : i + chunk_size] if t.shape[0] != 1 else t
            )
        else:
            select_chunk = (
                partial(
                    _chunk_slice, 
                    flat_start=i, 
                    flat_end=min(flat_batch_dim, i + chunk_size), 
                    no_batch_dims=len(orig_batch_dims)
                )
            )

        chunks = tensor_tree_map(select_chunk, prepped_inputs)

        # Run the layer on the chunk
        output_chunk = layer(**chunks)

        # Allocate space for the output
        if out is None:
            allocate = lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:])
            out = tensor_tree_map(allocate, output_chunk)

        # Put the chunk in its pre-allocated space
        out_type = type(output_chunk)
        if out_type is dict:
            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    else:
                        v[i : i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                x1[i : i + chunk_size] = x2
        elif out_type is torch.Tensor:
            out[i : i + chunk_size] = output_chunk
        else:
            raise ValueError("Not supported")

        i += chunk_size

    reshape = lambda t: t.view(orig_batch_dims + t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out
