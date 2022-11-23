import torch

from colossalai.fx import symbolic_trace

try:
    from torchrec.models import dlrm
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
    NOT_TORCHREC = False
except ImportError:
    NOT_TORCHREC = True

import pytest

BATCH = 2
SHAPE = 10


@pytest.mark.skipif(NOT_TORCHREC, reason='torchrec is not installed')
def test_torchrec_dlrm_models():
    MODEL_LIST = [
        dlrm.DLRM,
        dlrm.DenseArch,
        dlrm.InteractionArch,
        dlrm.InteractionV2Arch,
        dlrm.OverArch,
        dlrm.SparseArch,
    # dlrm.DLRMV2
    ]

    # Data Preparation
    # EmbeddingBagCollection
    eb1_config = EmbeddingBagConfig(name="t1", embedding_dim=SHAPE, num_embeddings=SHAPE, feature_names=["f1"])
    eb2_config = EmbeddingBagConfig(name="t2", embedding_dim=SHAPE, num_embeddings=SHAPE, feature_names=["f2"])

    ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
    keys = ["f1", "f2"]

    # KeyedTensor
    KT = KeyedTensor(keys=keys, length_per_key=[SHAPE, SHAPE], values=torch.rand((BATCH, 2 * SHAPE)))

    # KeyedJaggedTensor
    KJT = KeyedJaggedTensor.from_offsets_sync(keys=keys,
                                              values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                                              offsets=torch.tensor([0, 2, 4, 6, 8]))

    # Dense Features
    dense_features = torch.rand((BATCH, SHAPE))

    # Sparse Features
    sparse_features = torch.rand((BATCH, len(keys), SHAPE))

    for model_cls in MODEL_LIST:
        # Initializing model
        if model_cls == dlrm.DLRM:
            model = model_cls(ebc, SHAPE, [SHAPE, SHAPE], [5, 1])
        elif model_cls == dlrm.DenseArch:
            model = model_cls(SHAPE, [SHAPE, SHAPE])
        elif model_cls == dlrm.InteractionArch:
            model = model_cls(len(keys))
        elif model_cls == dlrm.InteractionV2Arch:
            I1 = dlrm.DenseArch(3 * SHAPE, [3 * SHAPE, 3 * SHAPE])
            I2 = dlrm.DenseArch(3 * SHAPE, [3 * SHAPE, 3 * SHAPE])
            model = model_cls(len(keys), I1, I2)
        elif model_cls == dlrm.OverArch:
            model = model_cls(SHAPE, [5, 1])
        elif model_cls == dlrm.SparseArch:
            model = model_cls(ebc)
        elif model_cls == dlrm.DLRMV2:
            # Currently DLRMV2 cannot be traced
            model = model_cls(ebc, SHAPE, [SHAPE, SHAPE], [5, 1], [4 * SHAPE, 4 * SHAPE], [4 * SHAPE, 4 * SHAPE])

        # Setup GraphModule
        if model_cls == dlrm.InteractionV2Arch:
            concrete_args = {"dense_features": dense_features, "sparse_features": sparse_features}
            gm = symbolic_trace(model, concrete_args=concrete_args)
        else:
            gm = symbolic_trace(model)

        model.eval()
        gm.eval()

        # Aligned Test
        with torch.no_grad():
            if model_cls == dlrm.DLRM or model_cls == dlrm.DLRMV2:
                fx_out = gm(dense_features, KJT)
                non_fx_out = model(dense_features, KJT)
            elif model_cls == dlrm.DenseArch:
                fx_out = gm(dense_features)
                non_fx_out = model(dense_features)
            elif model_cls == dlrm.InteractionArch or model_cls == dlrm.InteractionV2Arch:
                fx_out = gm(dense_features, sparse_features)
                non_fx_out = model(dense_features, sparse_features)
            elif model_cls == dlrm.OverArch:
                fx_out = gm(dense_features)
                non_fx_out = model(dense_features)
            elif model_cls == dlrm.SparseArch:
                fx_out = gm(KJT)
                non_fx_out = model(KJT)

        if torch.is_tensor(fx_out):
            assert torch.allclose(
                fx_out, non_fx_out), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'
        else:
            assert torch.allclose(
                fx_out.values(),
                non_fx_out.values()), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


if __name__ == "__main__":
    test_torchrec_dlrm_models()
