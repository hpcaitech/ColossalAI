import pytest
import torch

from colossalai.fx import symbolic_trace

try:
    from torchrec.models import deepfm
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
    NOT_TORCHREC = False
except ImportError:
    NOT_TORCHREC = True

BATCH = 2
SHAPE = 10


@pytest.mark.skipif(NOT_TORCHREC, reason='torchrec is not installed')
def test_torchrec_deepfm_models():
    MODEL_LIST = [deepfm.DenseArch, deepfm.FMInteractionArch, deepfm.OverArch, deepfm.SimpleDeepFMNN, deepfm.SparseArch]

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
    features = torch.rand((BATCH, SHAPE))

    for model_cls in MODEL_LIST:
        # Initializing model
        if model_cls == deepfm.DenseArch:
            model = model_cls(SHAPE, SHAPE, SHAPE)
        elif model_cls == deepfm.FMInteractionArch:
            model = model_cls(SHAPE * 3, keys, SHAPE)
        elif model_cls == deepfm.OverArch:
            model = model_cls(SHAPE)
        elif model_cls == deepfm.SimpleDeepFMNN:
            model = model_cls(SHAPE, ebc, SHAPE, SHAPE)
        elif model_cls == deepfm.SparseArch:
            model = model_cls(ebc)

        # Setup GraphModule
        gm = symbolic_trace(model)

        model.eval()
        gm.eval()

        # Aligned Test
        with torch.no_grad():
            if model_cls == deepfm.DenseArch or model_cls == deepfm.OverArch:
                fx_out = gm(features)
                non_fx_out = model(features)
            elif model_cls == deepfm.FMInteractionArch:
                fx_out = gm(features, KT)
                non_fx_out = model(features, KT)
            elif model_cls == deepfm.SimpleDeepFMNN:
                fx_out = gm(features, KJT)
                non_fx_out = model(features, KJT)
            elif model_cls == deepfm.SparseArch:
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
    test_torchrec_deepfm_models()
