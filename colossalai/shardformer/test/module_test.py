import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import colossalai
from colossalai.shardformer.layer.dist_crossentropy import applyDistCrossEntropy
from colossalai.shardformer.layer.dropout import Dropout1D


def get_args():
    parser = colossalai.get_default_parser()
    parser.add_argument("--module", type=str, default='distloss')
    return parser.parse_args()


def test_dist_crossentropy():
    pred = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.randint(8, (1, 4)).repeat(2, 1)

    pred_ = pred.view(-1, 8)
    labels_ = labels.view(-1)
    loss = F.cross_entropy(pred_, labels_)
    loss.backward()
    print(f"normal loss:{loss}")

    pred = pred.chunk(int(os.environ['WORLD_SIZE']), -1)[int(os.environ['RANK'])]
    loss = applyDistCrossEntropy(pred.to('cuda'), labels.to('cuda'))
    loss.backward()
    print(f"dist loss:{loss}")


def test_dropout():
    input = torch.randn(5, 4).to("cuda")
    m = Dropout1D(p=0.2).to("cuda")
    for i in range(2):
        print(f"Output: {m(input)}")
        print(torch.randn(1))


if __name__ == '__main__':
    args = get_args()
    colossalai.launch_from_torch(config={})
    if args.module == 'distloss':
        test_dist_crossentropy()
    elif args.module == 'dropout':
        test_dropout()
    else:
        print("not implemented yet")
