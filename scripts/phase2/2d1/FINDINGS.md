# 2D1 可用性侦察：CPU tiny 检查记录

最新结论见 [REPORT.md](REPORT.md)。本文件保留各轮运行的历史范围和失败证据，最新进展在末节。

日期：2026-10-26。环境：开发容器 wzj-colossalai-dev-tf517，Python 3.10，Torch 2.13.0+cu130，Transformers 5.17.0。代码位于服务器现有主仓的 scripts/phase2/2d1/，主仓 origin 为 https://github.com/hpcaitech/ColossalAI.git，当时分支 feat/upgrade_transformer_5.17。

这是 L0 的阶段性运行证据，不代表 L1/L2 或完整 2D1 已验收。

## 已验证结果

| 模型 | 官方实现的 tiny 创建/前向 | 开启诊断适配后的结果 | 额外检查 |
| --- | --- | --- | --- |
| LLaDA-8B-Base | 通过；98,624 个参数 | 无需适配 | 固定掩码测试损失的反向与一步更新通过，21 个参数张量有有限非零梯度并更新 |
| LLaDA2.1-mini | 通过；96,896 个参数 | 保持原实现，通过 | MoE 层实际调用 1 次 |
| LLaDA2.2-mini | 通过；96,896 个参数 | 保持原实现，通过 | MoE 层实际调用 1 次 |
| Ouro-1.4B | 创建失败：KeyError: 'default' | 创建、前向通过；98,945 个参数 | 两层各执行两轮；因果前缀差值 0；10-28 加权 labels 分支反向与一步更新通过（27 个参数张量） |
| Ouro-2.6B | 创建失败：KeyError: 'default' | 创建、前向通过；98,945 个参数 | 两层各执行两轮；因果前缀差值 0 |

四个新模型的输入为两条长度 16 的随机 token 序列；词表缩小为 128，输出形状均为 (2, 16, 128)。这是各仓库代码的缩小配置测试，不是原始 1.4B/2.6B/16B 模型的真实权重测试。

LLaDA2.X 使用 2 层（1 dense + 1 MoE）、隐藏维度 64、4 个 attention heads、2 个 KV heads、4 个路由专家（每个 token 选 2 个）、1 个共享专家、2 个路由组。2.2 的 block_size 和 expert_capacity 均设为 4。Ouro 使用 2 层、隐藏维度 64、4 个 heads、2 轮循环。

## 固定版本与官方实现入口

| 仓库 | snapshot commit | 配置 / 模型代码 |
| --- | --- | --- |
| GSAI-ML/LLaDA-8B-Base | 0f2787f2d87eac5eed8a087d5ecd24277e6255b2 | configuration_llada.py / modeling_llada.py |
| inclusionAI/LLaDA2.1-mini | 20e64e2ad21644d0e5248586ed9c942cdd45de0f | configuration_llada2_moe.py / modeling_llada2_moe.py |
| inclusionAI/LLaDA2.2-mini | e4f77d25ab5ca86f2223c6cdc6a72b887e20c8a5 | configuration_llada2_moe.py / modeling_llada2_moe.py |
| ByteDance/Ouro-1.4B | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | configuration_ouro.py / modeling_ouro.py |
| ByteDance/Ouro-2.6B | 1ed04250da1a9936042725d302e81c8fa2ab5abd | configuration_ouro.py / modeling_ouro.py |

官方文件位于 /mnt/beegfs/ColossalAI/shared-cache/hub/models--组织--模型/snapshots/commit/。动态导入时复制到个人 HF modules 缓存的文件不是正式开发修改位置。

## Ouro 的三个阻塞

### OURO-01：默认 RoPE 初始化入口

原实现 modeling_ouro.py:459 使用 ROPE_INIT_FUNCTIONS[self.rope_type]，默认类型为 default。安装的 Transformers 5.17 注册表不再提供这个键，因此构造 OuroRotaryEmbedding 时抛出 KeyError。

这不表示 Transformers 不支持默认 RoPE，而是默认计算入口转到了模型实现中。诊断适配复用当前 Qwen2RotaryEmbedding.compute_default_rope_parameters，将它加入 Ouro 模块自己的注册表副本；不更改 Transformers 全局注册表。

### OURO-02：RoPE 参数初始化协议

只处理 OURO-01 后，OuroModel.post_init() 仍失败。Transformers modeling_utils.py:2345–2355 对 RotaryEmbedding 模块执行初始化时，会读取模块的 compute_default_rope_parameters；原 Ouro 类缺少该方法，得到 None，随后调用导致 TypeError: 'NoneType' object is not callable。

诊断适配临时把同一个 HF 默认 RoPE 计算函数挂为 OuroRotaryEmbedding 的静态方法，退出后恢复。这两处使用一致的计算函数。

### OURO-03：注意力掩码函数参数变化

处理前两个问题后，模型创建成功，前向在 modeling_ouro.py:577 失败：
TypeError: create_causal_mask() got an unexpected keyword argument 'input_embeds'。

原调用传入 input_embeds 和 cache_position；当前 5.17 的签名使用 inputs_embeds，且不接收 cache_position。诊断包装仅在无 KV cache、位置从 0 连续递增、标准 position_ids 的条件下重命名参数并不再传入 cache_position；超出此范围明确报错，不声称支持缓存解码或其他位置方案。

进一步对两个 Ouro 做因果性检查：改变序列后 8 个 token，比较前 8 个位置的 logits，最大绝对差均为 0。这能发现明显的未来信息泄漏，不等于与旧版官方实现完成数值对齐。

## 日志与 JSON 证据

以下均为容器中的持久化路径，宿主机对应开发容器目录下 results/2d1/：

- LLaDA 前向回归：/results/2d1/llada-tiny-20261026-070417-vqjGty.log
- LLaDA 反向与更新：/results/2d1/llada-backward-20261026-070421-0RWvCu.log
- 四模型原始基线：/results/2d1/remote-tiny-20261026-070853-kQT5Wn.log
- 原始结果 JSON：/results/2d1/remote-tiny-20261026-070857-942.json
- 仅补注册入口，暴露初始化协议问题：/results/2d1/remote-tiny-20261026-071143-aJ4Y3e.log
- 补齐 RoPE 两处，暴露 mask 参数问题：/results/2d1/remote-tiny-20261026-071348-JnYcxN.log
- 完整诊断适配：/results/2d1/remote-tiny-20261026-071451-d0QHoC.log
- 完整诊断结果 JSON：/results/2d1/remote-tiny-rope-mask-20261026-071455-1214.json

第一份部分适配日志中的 rope 标签是在只补注册入口时运行的历史版本；现在使用 --compat rope-registry 可复现该阶段。各日志包含当时脚本的 SHA256，区分脚本版本。

## 已知非阻塞提示

- LLaDA2 默认 vocab_size=30592 与 pad_token_id=126081 不一致，配置打印时构造默认对象会触发警告。LLaDA2.1 实际 checkpoint 配置为 vocab_size=157184、pad_token_id=156892，有效；tiny 参数显式设置有效 ID。
- Ouro 的 rope_config_validation、layer_type_validation、check_model_inputs 弃用提示，以及 cache_position 的 docstring 检查信息仍保留。它们与上述实际异常分开记录。

## 代码落点与下一步

- 当前侦察入口：scripts/phase2/2d1/check_remote_tiny.py。
- 当前诊断适配：scripts/phase2/2d1/ouro_compat_probe.py，仅验证修复方向。
- 正式支持应继续复用官方模型实现。建议将 Ouro 创建前必需的兼容逻辑放入项目维护的专用适配模块，例如 colossalai/shardformer/modeling/ouro.py；该文件尚未新增，具体接口需结合现有兼容层设计确认。
- 模型创建成功后的接入规则再放入 colossalai/shardformer/policies/ 下，并以 tp_size=1 开始。仅靠创建模型之后执行的 policy 无法解决这次构造阶段错误。
- tiny 配置成熟后再接入 tests/kit/model_zoo/，保留原始失败复现，增加正式适配的回归测试。

尚未覆盖：LLaDA2/Ouro-2.6B 反向与参数更新、Ouro 循环参数的梯度数值正确性、正式扩散训练目标、GPU/BF16、真实权重加载、tokenizer/生成/KV cache/padding、checkpoint、DP/ZeRO/Gemini、与官方参考实现的数值对齐、收敛或性能。完整 L0 还需要补齐官方训练代码位置、版本支持范围与正式接入设计等记录。


## 2026-10-28：Ouro-1.4B tiny 训练检查

脚本：/workspace/ColossalAI/scripts/phase2/2d1/check_ouro_backward.py。
默认命令：bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh ouro-backward。
仍然使用上文固定的 Ouro-1.4B 快照、98945 个随机初始化参数、两层、两轮循环，
CPU / FP32 / eager / train / 无 KV cache，seed=42。

本次使用官方模型的 labels 训练分支：根据 early_exit_gate 的退出概率加权各轮 logits，
再计算 next-token loss。官方 modeling_ouro.py:762–772 是该分支，
不是仅对最后一轮输出手工添加损失。退出门的 weight 和 bias 因此都应获得梯度。

实际结果：
- BUILD_OK / FORWARD_OK，输出形状 (2, 16, 128)。
- LOSS_OK：4.816987037658691；手工移位交叉熵也是 4.816987037658691，差值为 0。
- 两个 Decoder 层的执行轮次均为 [0, 1]，四个中间输出都有有限非零梯度。
- 27 个可训练参数张量均有有限非零梯度，没有缺失或无效梯度。
- early_exit_gate.weight 梯度范数 0.04216048866510391；
  early_exit_gate.bias 梯度范数 0.002511051017791033。
- AdamW(lr=1e-3, weight_decay=0) 更新了全部 27 个参数张量，更新后参数全部有限。
- 适配器退出后原始绑定恢复；运行退出码为 0。
- 关闭适配的对照运行在 build 阶段复现 KeyError: 'default'，通过 run.sh 返回退出码 1，
  原始失败并未被日志管道掩盖。

证据：
- 通过日志：/results/2d1/ouro-backward-20261028-034537-BA6tE3.log
- 通过 JSON：/results/2d1/ouro-backward-ouro14-rope-mask-20261028-034540-1357.json
- 原始失败日志：/results/2d1/ouro-backward-20261028-034559-FfoRvc.log
- 原始失败 JSON：/results/2d1/ouro-backward-ouro14-none-20261028-034602-1460.json

LOSS_ALIGNMENT_OK 仅核对给定 logits 的 next-token 交叉熵，并非旧版官方模型的数值对齐。
中间输出收到梯度能确认两轮进入反向图，不证明循环共享参数的累计梯度在数值上完全正确。
本次没有新增正式 ColossalAI 模型适配或 policy，也未修改官方模型文件、共享缓存或 Transformers。
Ouro-2.6B 与 LLaDA2.1/2.2 的反向和更新仍待验证。

## 2026-10-28：2D1 探测收尾（新 main 分支）

本节更新前文“尚未覆盖”的状态：五模型的 tiny 梯度连通性和一步更新现在均已有证据；
正式建模适配、官方数值对齐、真实模型执行和 L1/L2 仍未完成。
报告入口为 [REPORT.md](REPORT.md)。本轮没有修改 colossalai/ 或 tests/，没有安装依赖或修改共享缓存。

### 最终运行

- 分支 feat/support_llada_and_ouro，HEAD d414d565d2002a8b4e79d810801860d8b0636aa7。
- 汇总日志：/results/2d1/suite-20261028-084707-AMNImD.log。
- 汇总 JSON：/results/2d1/suite-20261028-084707-3727/summary.json。
- 10 项检查均符合预期，suite 返回 0。remote-original 子命令仍返回 1，
  仅在两个 LLaDA2 前向通过、两个 Ouro 精确在 build 阶段报 KeyError('default') 时视为预期被满足。
- 子报告 assets.json、remote-original.json、remote-adapted.json、
  llada21-backward.json、llada22-backward.json、ouro14-backward.json、ouro26-backward.json 位于同一个汇总目录；
  每个子命令还有独立的 *.stdout.log 和 run.sh 持久化日志。
- check_imports.py 与 check_llada_tiny.py 现也锁定完整快照 SHA，避免 refs/main 漂移。

### LLaDA2 训练路径

新脚本 check_llada2_backward.py 不修改模型实现，使用官方 labels 分支。
将一半固定位置替换为 mask token，其余 labels 设为 -100；手工核对同位置 masked CE。
这不是随机扩散时间步、按 t 加权或完整 block-diffusion 训练目标。

两版本均为 96,896 参数、2 层（1 dense + 1 MoE）、4 个专家 top-2、1 个共享专家。
脚本验证 train 模式实际进入可微专家路径；推理分支中的 no_grad 不能作为训练通过的证据。
router、共享专家、参与监督位置的路由专家、attention、embedding、norm 和 head 均收到梯度并更新。
expert_bias 是 buffer，不将其当成可训练参数要求梯度。

- LLaDA2.1：loss 4.82357120513916；手工 CE 差约 4.77e-7；
  路由计数 [32,32,0,0]；监督位置计数 [16,16,0,0]；28 个非零梯度参数张量更新。
- LLaDA2.2：loss 4.819789409637451；
  路由计数 [32,20,6,6]；监督位置计数 [16,12,0,4]；31 个非零梯度参数张量更新。
  专家 2 接收 6 个未监督位置，最后一层 MoE 后仅有逐 token 的 norm/head，
  因而这些位置不影响损失，专家 2 的零梯度合法。脚本额外断言最后 MoE 输出的未监督位置梯度为零。
- 失败历史保留：suite-20261028-083848-1842 中脚本误用了 embedding 名称；
  suite-20261028-084204-2733 中对“所有接收过 token 的专家必须非零”要求过强。
  前者按官方 word_embeddings 名称修正；后者用真实监督位置的路由和输出梯度明确区分。
  两次均是新增探针的检查逻辑修正，没有更改官方模型。

### 资源与文件清单

| 模型 | 清单文件数 | 权重分片数 | header 存储元素数 | 权重 tensor 数据大小（十进制 GB） | config 保存版本 |
|---|---:|---:|---:|---:|---|
| LLaDA-8B | 16 | 6 | 8,015,581,184 | 16.031 | 4.46.3 |
| LLaDA2.1 | 17 | 8 | 16,255,643,392 | 32.511 | 4.57.1 |
| LLaDA2.2 | 20 | 7 | 16,255,643,392 | 32.511 | 5.2.0 |
| Ouro-1.4B | 13 | 1 | 1,434,652,673 | 2.869 | 4.55.0 |
| Ouro-2.6B | 13 | 1 | 2,667,974,657 | 5.336 | 4.55.0 |

check_assets.py 核对固定 revision 的 MANIFEST.tsv、全部文件大小、当前非权重 SHA256、
safetensors header 中的 shape/dtype/offset/文件结尾、分片索引键集合与总字节数；
随后离线加载 tokenizer，检查中英文编码/解码和词表编号范围。
未重新扫描全部权重字节；23 个大分片沿用维护者已有 hub_check=ok 的哈希记录。
需要重新读取全部权重时可显式运行 assets --full-sha256，约涉及 89 GB 共享存储读取。
79 个清单文件中没有独立 LICENSE 文件；许可按对应官方模型卡记录，未作额外法律判断。
“文件就绪”不代表已执行完整 from_pretrained 大模型加载。

### GPU / 内存条件

只读脚本 inspect_gpu_access.sh 在宿主机运行，日志：
/results/2d1/gpu-access-20261028-164433-mLS7DZ.log（宿主机持久化目录中的同名文件）。
宿主机 nvidia-smi 返回 0，识别 8 张 H20，每张 97871 MiB；
容器 nvidia-smi 返回 255：Failed to initialize NVML: Unknown Error。
docker 配置仍申请了全部 GPU，但本次没有重启容器、修改 GPU 状态或驱动。
容器内存限额 17,179,869,184 字节（16 GiB），CPU 配额 4 核；
宿主机约 2 TiB RAM 并不等于容器可以使用这些内存。

后续真实模型训练前处理容器 GPU 访问；CPU offload 还需更高容器内存额度。
按任务文档的近似 4 字节/参数 GPU + 12 字节/参数 CPU 状态预算，
8B 可先规划至少 128 GiB 主机内存额度，16B 可先规划至少 256 GiB，
实际需求再加数据、激活、构造/保存峰值，并在 L1 实测。
本轮只给出 batch=1、seq=128、BF16 的真实模型起步候选，没有声称测出最大序列长度。

### 版本与训练参考边界

最低支持版本和 config 中的 transformers_version 是不同信息。
本轮精确验证 5.17.0；没有跨旧版扫描，不能据此声称 >=5.17 或整个 v5 都可用。
官方 LLaDA README 的 4.38.2、Ouro 模型卡的 <4.56 / 推荐 4.54.1 均按参考环境记录。
LLaDA2 的保存版本 4.57.1 / 5.2.0 不等于其最低版本；公开材料没有提供足够明确的下界。
Ouro 同一模型卡同时保留旧版建议和新版 cache 修复说明，固定快照的实测结果优先。

训练来源链接与结论见 REPORT.md：LLaDA 的 GUIDELINES 是核心目标参考；
dFactory 是 LLaDA2 的官方关联框架，但支持表不保证 2.1/2.2 完整覆盖；
Ouro 的 HF labels 分支可作本轮计算图参考，不能等同完整预训练目标。
