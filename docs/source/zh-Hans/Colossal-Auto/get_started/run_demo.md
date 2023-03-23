# 快速上手

Colossal-AI 提供了业界急需的一套高效易用自动并行系统。相比现有其他手动配置复杂并行策略和修改模型的解决方案，Colossal-AI 仅需增加一行代码，提供 cluster 信息以及单机训练模型即可获得分布式训练能力。Colossal-Auto的快速上手示例如下。

### 1. 基本用法
Colossal-Auto 可被用于为每一次操作寻找一个包含数据、张量（如1D、2D、序列化）的混合SPMD并行策略。您可参考[GPT 示例](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt/experiments/auto_parallel)。
详细的操作指引见其 `README.md`。

### 2. 与 activation checkpoint 结合

作为大模型训练中必不可少的显存压缩技术，Colossal-AI 也提供了对于 activation checkpoint 的自动搜索功能。相比于大部分将最大显存压缩作为目标的技术方案，Colossal-AI 的搜索目标是在显存预算以内，找到最快的 activation checkpoint 方案。同时，为了避免将 activation checkpoint 的搜索一起建模到 SPMD solver 中导致搜索时间爆炸，Colossal-AI 做了 2-stage search 的设计，因此可以在合理的时间内搜索到有效可行的分布式训练方案。 您可参考 [Resnet 示例](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/auto_parallel)。
详细的操作指引见其 `README.md`。
