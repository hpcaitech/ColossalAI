from curses import meta
from math import dist
from xml.dom import HierarchyRequestErr
from colossalai.fx.tracer import meta_patch
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.fx.tracer.meta_patch.patched_function import python_ops
import torch
from torchrec.sparse.jagged_tensor import KeyedTensor, KeyedJaggedTensor
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.models import deepfm, dlrm
import colossalai.fx as fx
import pdb
from torch.fx import GraphModule

BATCH = 4
SHAPE = 100


def ColoTracerTest(M: torch.nn.Module, meta_args: dict = None, concrete_args: dict = None) -> None:
    tracer = fx.ColoTracer()
    if not meta_args:
        meta_args = {}
    if not concrete_args:
        concrete_args = {}
    try:
        graph = tracer.trace(M, meta_args=meta_args, concrete_args=concrete_args)
    except:
        raise Exception(f"{M._get_name()} tracing failed!")
    return graph
    # print(f"{M._get_name()} tracing success!")
    # print(graph)


# print("============================Start testing deepfm============================")
# Test deepfm.DenseArch
B = 20
D = 3
in_features = 10
M = deepfm.DenseArch(in_features=10, hidden_layer_size=10, embedding_dim=D)
graph = ColoTracerTest(M)
gm = GraphModule(M, graph, M.__class__.__name__)
gm.recompile()

M.eval()
gm.eval()
B = 20
D = 3
data = torch.rand((B, 10))
with torch.no_grad():
    fx_out = gm(data)
    non_fx_out = M(data)
assert torch.allclose(fx_out, non_fx_out), f'{M.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'

# Test deepfm.FMInteractionArch
keys = ["f1", "f2"]
F = len(keys)
D = 3
B = 10
M = deepfm.FMInteractionArch(sparse_feature_names=keys, fm_in_features=D * D, deep_fm_dimension=D)
graph = ColoTracerTest(M)
gm = GraphModule(M, graph, M.__class__.__name__)
gm.recompile()

M.eval()
gm.eval()

keys = ["f1", "f2"]
F = len(keys)
dense_features = torch.rand((B, D))
sparse_features = KeyedTensor(
    keys=keys,
    length_per_key=[D, D],
    values=torch.rand((B, D * F)),
)
with torch.no_grad():
    fx_out = gm(dense_features, sparse_features)
    non_fx_out = M(dense_features, sparse_features)
assert torch.allclose(fx_out, non_fx_out), f'{M.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'

# Test deepfm.OverArch
B = 20
M = deepfm.OverArch(in_features=10)
graph = ColoTracerTest(M)

gm = GraphModule(M, graph, M.__class__.__name__)
gm.recompile()

M.eval()
gm.eval()

data = torch.rand((B, 10))
with torch.no_grad():
    fx_out = gm(data)
    non_fx_out = M(data)
assert torch.allclose(fx_out, non_fx_out), f'{M.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'

# Test deepfm.SimpleDeepFMNN
eb1_config = EmbeddingBagConfig(name="t1", embedding_dim=SHAPE, num_embeddings=SHAPE, feature_names=["f1", "f3"])
eb2_config = EmbeddingBagConfig(
    name="t2",
    embedding_dim=SHAPE,
    num_embeddings=SHAPE,
    feature_names=["f2"],
)

B = 2

eb1_config = EmbeddingBagConfig(name="t1", embedding_dim=SHAPE, num_embeddings=SHAPE, feature_names=["f1", "f3"])
eb2_config = EmbeddingBagConfig(
    name="t2",
    embedding_dim=SHAPE,
    num_embeddings=SHAPE,
    feature_names=["f2"],
)

ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
M = deepfm.SimpleDeepFMNN(embedding_bag_collection=ebc,
                          num_dense_features=SHAPE,
                          hidden_layer_size=SHAPE,
                          deep_fm_dimension=SHAPE)

features = torch.rand((B, SHAPE))
sparse_features = KeyedJaggedTensor.from_offsets_sync(
    keys=["f1", "f2", "f3"],
    values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 10, 11, 12, 23]),
    offsets=torch.tensor([0, 2, 4, 6, 8, 10, 12]),
)
graph = ColoTracerTest(M)

gm = GraphModule(M, graph, M.__class__.__name__)
gm.recompile()

M.eval()
gm.eval()

data = torch.rand((B, 10))
with torch.no_grad():
    fx_out = gm(features, sparse_features)
    non_fx_out = M(features, sparse_features)
assert torch.allclose(fx_out, non_fx_out), f'{M.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'

# Test deepfm.SparseArch
eb1_config = EmbeddingBagConfig(name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"])
eb2_config = EmbeddingBagConfig(name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"])

ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

features = KeyedJaggedTensor.from_offsets_sync(
    keys=["f1", "f2"],
    values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
    offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
)

M = deepfm.SparseArch(ebc)
graph = ColoTracerTest(M)

gm = GraphModule(M, graph, M.__class__.__name__)
gm.recompile()

M.eval()
gm.eval()

with torch.no_grad():
    fx_out = gm(features)
    non_fx_out = M(features)
assert torch.allclose(fx_out._values,
                      non_fx_out._values), f'{M.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'

# print("============================Start testing dlrm============================")
# Test dlrm.DLRM
# M = dlrm.DLRM(embedding_bag_collection=ebc,
#               dense_in_features=SHAPE,
#               dense_arch_layer_sizes=[SHAPE],
#               over_arch_layer_sizes=[5, 1])
# ColoTracerTest(M)

    # # Test dlrm.DLRMTrain
    # M = dlrm.DLRMTrain(M)
    # ColoTracerTest(M)

    # # Test dlrm.DLRMV2
    # """
    # Currently there are some problems concerning KeyedJaggedTensor,
    # so I comment the DLRMV2 test
    # """
    # # B = 2
    # # D = 8

    # # eb1 = EmbeddingBagConfig(
    # #    name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
    # # )
    # # eb2 = EmbeddingBagConfig(
    # #    name="t2",
    # #    embedding_dim=D,
    # #    num_embeddings=100,
    # #    feature_names=["f2"],
    # # )

    # # ebc2 = EmbeddingBagCollection(tables=[eb1, eb2])
    # # M = dlrm.DLRMV2(embedding_bag_collection=ebc2,
    # #                 dense_in_features=100,
    # #                 dense_arch_layer_sizes=[20, D],
    # #                 over_arch_layer_sizes=[5, 1],
    # #                 interaction_branch1_layer_sizes=[4 * D, 4 * D],
    # #                 interaction_branch2_layer_sizes=[4 * D, 4 * D])

    # # meta_args = {"dense_features": torch.rand(B, SHAPE, device="meta"),
    # #              "sparse_features": KeyedJaggedTensor.from_offsets_sync(
    # #              keys=["f1", "f2", "f3"],
    # #              values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 10, 12, 24, 35], device="meta"),
    # #              offsets=torch.tensor([0, 2, 4, 6, 8, 10, 12]),)}
    # # pdb.set_trace()
    # # ColoTracerTest(M, meta_args=meta_args)
    # # ColoTracerTest(M, meta_args=meta_args)
    # # ColoTracerTest(M, concrete_args=concrete_args)
    # # gm = tracer.Tracer().trace(M, concrete_args=concrete_args)

    # # Test dlrm.DenseArch
    # M = dlrm.DenseArch(in_features=SHAPE,
    #                    layer_sizes=[SHAPE, SHAPE])
    # ColoTracerTest(M)

    # # Test dlrm.InteractionArch
    # M = dlrm.InteractionArch(num_sparse_features=SHAPE)
    # ColoTracerTest(M)

    # # Test dlrm.InteractionV2Arch
    # I1 = dlrm.DenseArch(
    #     in_features= 4 * SHAPE,
    #     layer_sizes=[4*SHAPE, 4*SHAPE], # F1 = 4
    # )
    # I2 = dlrm.DenseArch(
    #     in_features= 3 * SHAPE + SHAPE,
    #     layer_sizes=[4*SHAPE, 4*SHAPE], # F2 = 4
    # )
    # M = dlrm.InteractionV2Arch(num_sparse_features=SHAPE,
    #                            interaction_branch1=I1,
    #                            interaction_branch2=I2,
    #                            )
    # # meta_args = {"dense_features": torch.rand(SHAPE, SHAPE, device="meta"), "sparse_features": torch.rand(SHAPE, F, SHAPE, device='meta')}
    # concrete_args = {"dense_features": torch.rand(SHAPE, SHAPE), "sparse_features": torch.rand(SHAPE, F, SHAPE)}
    # ColoTracerTest(M, concrete_args=concrete_args)

    # # Test dlrm.OverArch
    # M = dlrm.OverArch(in_features=SHAPE, layer_sizes=[5, 1])
    # ColoTracerTest(M)

    # # Test dlrm.SparseArch
    # M = dlrm.SparseArch(ebc)
    # ColoTracerTest(M)
