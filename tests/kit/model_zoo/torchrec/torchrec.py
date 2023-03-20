from collections import namedtuple
from functools import partial

import torch

try:
    from torchrec.models import deepfm, dlrm
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
    NO_TORCHREC = False
except ImportError:
    NO_TORCHREC = True

from ..registry import ModelAttribute, model_zoo


def register_torchrec_models():
    BATCH = 2
    SHAPE = 10
    # KeyedTensor
    KT = KeyedTensor(keys=["f1", "f2"], length_per_key=[SHAPE, SHAPE], values=torch.rand((BATCH, 2 * SHAPE)))

    # KeyedJaggedTensor
    KJT = KeyedJaggedTensor.from_offsets_sync(keys=["f1", "f2"],
                                              values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                                              offsets=torch.tensor([0, 2, 4, 6, 8]))

    data_gen_fn = lambda: dict(features=torch.rand((BATCH, SHAPE)))

    interaction_arch_data_gen_fn = lambda: dict(dense_features=torch.rand((BATCH, SHAPE)), sparse_features=KT)

    simple_dfm_data_gen_fn = lambda: dict(dense_features=torch.rand((BATCH, SHAPE)), sparse_features=KJT)

    sparse_arch_data_gen_fn = lambda: dict(features=KJT)

    output_transform_fn = lambda x: dict(output=x)

    def get_ebc():
        # EmbeddingBagCollection
        eb1_config = EmbeddingBagConfig(name="t1", embedding_dim=SHAPE, num_embeddings=SHAPE, feature_names=["f1"])
        eb2_config = EmbeddingBagConfig(name="t2", embedding_dim=SHAPE, num_embeddings=SHAPE, feature_names=["f2"])
        return EmbeddingBagCollection(tables=[eb1_config, eb2_config])

    model_zoo.register(name='deepfm_densearch',
                       model_fn=partial(deepfm.DenseArch, SHAPE, SHAPE, SHAPE),
                       data_gen_fn=data_gen_fn,
                       output_transform_fn=output_transform_fn)

    model_zoo.register(name='deepfm_interactionarch',
                       model_fn=partial(deepfm.FMInteractionArch, SHAPE * 3, ["f1", "f2"], SHAPE),
                       data_gen_fn=interaction_arch_data_gen_fn,
                       output_transform_fn=output_transform_fn)

    model_zoo.register(name='deepfm_overarch',
                       model_fn=partial(deepfm.OverArch, SHAPE),
                       data_gen_fn=data_gen_fn,
                       output_transform_fn=output_transform_fn)

    model_zoo.register(name='deepfm_simpledeepfmnn',
                       model_fn=partial(deepfm.SimpleDeepFMNN, SHAPE, get_ebc(), SHAPE, SHAPE),
                       data_gen_fn=simple_dfm_data_gen_fn,
                       output_transform_fn=output_transform_fn)

    model_zoo.register(name='deepfm_sparsearch',
                       model_fn=partial(deepfm.SparseArch, get_ebc()),
                       data_gen_fn=sparse_arch_data_gen_fn,
                       output_transform_fn=output_transform_fn)

    model_zoo.register(name='dlrm',
                       model_fn=partial(dlrm.DLRM, get_ebc(), SHAPE, [SHAPE, SHAPE], [5, 1]),
                       data_gen_fn=simple_dfm_data_gen_fn,
                       output_transform_fn=output_transform_fn)

    model_zoo.register(name='dlrm_densearch',
                       model_fn=partial(dlrm.DenseArch, SHAPE, [SHAPE, SHAPE]),
                       data_gen_fn=data_gen_fn,
                       output_transform_fn=output_transform_fn)

    model_zoo.register(name='dlrm_interactionarch',
                       model_fn=partial(dlrm.InteractionArch, 2),
                       data_gen_fn=interaction_arch_data_gen_fn,
                       output_transform_fn=output_transform_fn)

    model_zoo.register(name='dlrm_overarch',
                       model_fn=partial(dlrm.OverArch, SHAPE, [5, 1]),
                       data_gen_fn=data_gen_fn,
                       output_transform_fn=output_transform_fn)

    model_zoo.register(name='dlrm_sparsearch',
                       model_fn=partial(dlrm.SparseArch, get_ebc()),
                       data_gen_fn=sparse_arch_data_gen_fn,
                       output_transform_fn=output_transform_fn)


if not NO_TORCHREC:
    register_torchrec_models()
