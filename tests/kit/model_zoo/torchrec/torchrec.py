from functools import partial

import torch
from torchrec.models import deepfm, dlrm
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

from ..registry import model_zoo

BATCH = 2
SHAPE = 10


def gen_kt():
    KT = KeyedTensor(keys=["f1", "f2"], length_per_key=[SHAPE, SHAPE], values=torch.rand((BATCH, 2 * SHAPE)))
    return KT


# KeyedJaggedTensor
def gen_kjt():
    KJT = KeyedJaggedTensor.from_offsets_sync(
        keys=["f1", "f2"], values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), offsets=torch.tensor([0, 2, 4, 6, 8])
    )
    return KJT


data_gen_fn = lambda: dict(features=torch.rand((BATCH, SHAPE)))


def interaction_arch_data_gen_fn():
    KT = gen_kt()
    return dict(dense_features=torch.rand((BATCH, SHAPE)), sparse_features=KT)


def simple_dfm_data_gen_fn():
    KJT = gen_kjt()
    return dict(dense_features=torch.rand((BATCH, SHAPE)), sparse_features=KJT)


def sparse_arch_data_gen_fn():
    KJT = gen_kjt()
    return dict(features=KJT)


def output_transform_fn(x):
    if isinstance(x, KeyedTensor):
        output = dict()
        for key in x.keys():
            output[key] = x[key]
        return output
    else:
        return dict(output=x)


def get_ebc():
    # EmbeddingBagCollection
    eb1_config = EmbeddingBagConfig(name="t1", embedding_dim=SHAPE, num_embeddings=SHAPE, feature_names=["f1"])
    eb2_config = EmbeddingBagConfig(name="t2", embedding_dim=SHAPE, num_embeddings=SHAPE, feature_names=["f2"])
    return EmbeddingBagCollection(tables=[eb1_config, eb2_config], device=torch.device("cpu"))


def sparse_arch_model_fn():
    ebc = get_ebc()
    return deepfm.SparseArch(ebc)


def simple_deep_fmnn_model_fn():
    ebc = get_ebc()
    return deepfm.SimpleDeepFMNN(SHAPE, ebc, SHAPE, SHAPE)


def dlrm_model_fn():
    ebc = get_ebc()
    return dlrm.DLRM(ebc, SHAPE, [SHAPE, SHAPE], [5, 1])


def dlrm_sparsearch_model_fn():
    ebc = get_ebc()
    return dlrm.SparseArch(ebc)


model_zoo.register(
    name="deepfm_densearch",
    model_fn=partial(deepfm.DenseArch, SHAPE, SHAPE, SHAPE),
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="deepfm_interactionarch",
    model_fn=partial(deepfm.FMInteractionArch, SHAPE * 3, ["f1", "f2"], SHAPE),
    data_gen_fn=interaction_arch_data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="deepfm_overarch",
    model_fn=partial(deepfm.OverArch, SHAPE),
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="deepfm_simpledeepfmnn",
    model_fn=simple_deep_fmnn_model_fn,
    data_gen_fn=simple_dfm_data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="deepfm_sparsearch",
    model_fn=sparse_arch_model_fn,
    data_gen_fn=sparse_arch_data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="dlrm", model_fn=dlrm_model_fn, data_gen_fn=simple_dfm_data_gen_fn, output_transform_fn=output_transform_fn
)

model_zoo.register(
    name="dlrm_densearch",
    model_fn=partial(dlrm.DenseArch, SHAPE, [SHAPE, SHAPE]),
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="dlrm_interactionarch",
    model_fn=partial(dlrm.InteractionArch, 2),
    data_gen_fn=interaction_arch_data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="dlrm_overarch",
    model_fn=partial(dlrm.OverArch, SHAPE, [5, 1]),
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="dlrm_sparsearch",
    model_fn=dlrm_sparsearch_model_fn,
    data_gen_fn=sparse_arch_data_gen_fn,
    output_transform_fn=output_transform_fn,
)
