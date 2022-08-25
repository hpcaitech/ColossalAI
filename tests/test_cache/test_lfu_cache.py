import colossalai
import torch
from colossalai.nn.parallel.layers.cache_embedding.freq_aware_embedding import FreqAwareEmbeddingBag
from colossalai.nn.parallel.layers.cache_embedding.cache_mgr import EvictionStrategy
import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# minimal test to check behavior
Bag = FreqAwareEmbeddingBag(
    5,
    5,
    cuda_row_num=3,
    buffer_size=0,
    pin_weight=True,
    warmup_ratio=0.0,
    ids_freq_mapping=torch.tensor([5,4,3,2,1],device="cuda:0"),
    evict_strategy=EvictionStrategy.LFU
)

offsets = torch.tensor([0],device="cuda:0")

# prepare frequency learning info:
Bag.forward(torch.tensor([0,1,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0,1,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0,1,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0,1,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0],device="cuda:0"),offsets)

# check strategy
Bag.forward(torch.tensor([0,1,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([3],device="cuda:0"),offsets)
Bag.forward(torch.tensor([2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([4],device="cuda:0"),offsets)
Bag.forward(torch.tensor([2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0],device="cuda:0"),offsets)

print(Bag.cache_weight_mgr.num_hits_history[-6:])
print(Bag.cache_weight_mgr.num_miss_history[-6:])