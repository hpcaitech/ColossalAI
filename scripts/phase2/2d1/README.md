# 2D1：模型兼容性侦察脚本

在开发容器 `wzj-colossalai-dev-tf517` 内运行。此目录挂载自服务器已有的 ColossalAI 工作副本，退出或重建容器后文件仍保留。

## 运行

在服务器上进入容器：

```bash
docker exec -it wzj-colossalai-dev-tf517 bash
```

在容器里执行五个模型的配置和代码导入检查：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh imports
```

在容器里执行 LLaDA 的 CPU tiny 模型创建和前向检查：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh llada-tiny
```

在容器里执行 LLaDA 的 CPU tiny 反向传播和一步参数更新检查：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh llada-backward
```

每次运行都会打印日志路径，并生成独立日志。默认目录为 `/results/2d1/`；可用环境变量 `D2D1_LOG_DIR` 指定其他可写目录。

服务器上的实际日志位置：
`/mnt/beegfs/ColossalAI/wangzhijian/containers/dev-torch213-tf517-peft0212/results/2d1/`。

## 检查范围

- `check_imports.py`：检查 LLaDA-8B、LLaDA2.1-mini、LLaDA2.2-mini、Ouro-1.4B、Ouro-2.6B；记录快照版本、实际词表和填充符编号。配置加载和类导入通过不代表模型能训练。
- `check_llada_tiny.py`：将内存中的 LLaDA 配置缩小为 2 层、隐藏维度 64、词表 128；随机初始化，在 CPU 上执行一次前向，检查输出形状和数值有限性。不加载原始权重，不执行反向或优化器更新。
- `check_llada_backward.py`：复用相同的小模型构建函数。随机 token 的固定一半位置替换为 mask token，仅在掩码位置计算交叉熵，检查 loss、全部可训练参数的梯度是否有限，并验证无权重衰减的 AdamW 确实更新参数。成功标记为 `LOSS_OK`、`BACKWARD_OK`、`STEP_OK`。这是梯度链路检查，不是 2D2 的正式扩散目标：不含随机时间步、按 t 加权或官方数值对齐。
- `run.sh`：记录时间、Python 路径、脚本 SHA256 和输出；强制 Hub 离线并隐藏 GPU。检查失败时返回非零退出码，保存日志不会掩盖失败。
- 使用模型仓库中的自定义 Python 代码；不安装依赖，不修改共享缓存原文件。
- 所有模型探测均锁定固定快照版本。模型计算脚本不加载完整大模型；assets 单独检查缓存文件、分片和 tokenizer，完整检查边界见下文。

## 查看结果

导入检查应逐个出现 `CONFIG_OK` 和 `IMPORT_OK`，最终汇总通过数量。tiny 检查应出现 `BUILD_OK`、输出形状 `(2, 16, 128)` 和 `FORWARD_OK`。

保留日志中的警告与异常。不要仅凭程序结束判断通过，也不要把前向通过当成训练正确性验收。

## 后续约定

同一任务的新侦察脚本继续放在此目录。其他第二阶段任务使用 `scripts/phase2/<任务编号>/`，并提供容器内运行命令与持久化日志位置。

## LLaDA2.X / Ouro 创建与前向检查

原始代码基线（当前两个 LLaDA2 通过、两个 Ouro 在创建阶段失败，退出码 1）：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh remote-tiny
```

仅检查一个模型（可选 llada21、llada22、ouro14、ouro26）：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh remote-tiny --model ouro14
```

开启 Ouro 的进程内诊断适配；对 LLaDA2 不应用适配：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh remote-tiny --compat rope-mask
```

`--compat` 默认为 `none`；`rope-registry` 只补默认 RoPE 注册入口，`rope` 同时补初始化要求的方法，`rope-mask` 进一步适配无 KV cache 的掩码调用。中间模式用于复现逐层暴露的问题，不应当作正式支持接口。

此检查固定四个模型的 snapshot commit，使用 CPU / FP32 / eager / eval / 无 KV cache。LLaDA2 保留一个 dense 层和一个 MoE 层（4 个路由专家、1 个共享专家）；Ouro 保留 2 层、2 轮循环。使用模块 hook 确认关键结构被执行，Ouro 还检查改变未来输入不会影响此前位置的输出。日志之外还生成 JSON 结果文件，路径在输出末尾显示。

`ouro_compat_probe.py` 仅为验证修复方向的实验，限制在 Transformers 5.17.0、default RoPE、连续位置、无缓存的 eager 前向。它在当前进程内临时替换导入模块的入口和类方法，结束后恢复；不修改官方源码文件或共享缓存，不修改 Transformers 全局 RoPE 注册表。尚未作为 ColossalAI 的正式适配层。

详细结果与证据见 [FINDINGS.md](FINDINGS.md)。


## Ouro tiny 训练检查（2026-10-28）

默认检查 Ouro-1.4B，启用现有的 rope-mask 实验适配：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh ouro-backward
```

新脚本 check_ouro_backward.py 复用 check_remote_tiny.py 的固定版本与 tiny 配置，
在 CPU / FP32 / eager / 无 KV cache 下创建随机小模型，执行一次训练更新。
可以使用 --model ouro14（默认）或 --model ouro26 选择模型；10-28 收尾运行已分别验证两个快照。
结果仍限于各自的 tiny 配置，不代表真实大模型训练已通过。

检查直接向官方模型传入 labels。该快照的 labels 分支会根据 early_exit_gate 的退出概率，
混合两轮 hidden states 对应的 logits，再计算下一个 token 的交叉熵；它不同于不传 labels 时的
最后一轮/阈值退出路径。脚本用 logits[:, :-1] 和 labels[:, 1:] 手工计算交叉熵，
核对标签移位与损失，随后对模型返回的 loss 执行 backward。

检查包括：
- 两个 Decoder 层分别执行第 0、1 轮；两轮输出都获得有限非零梯度。
- 这个固定的加权 labels 分支中，所有可训练参数都参与损失，要求它们的梯度存在且有限；
  对其他提前退出或自定义损失分支不能机械套用这一要求。
- embedding、每个 Decoder 层、最终 norm、lm_head 和 early_exit_gate 都有非零梯度并更新。
- AdamW 的 weight_decay=0，避免把权重衰减造成的变化误认为梯度学习。
- 适配一直保持到 backward/step 完成，退出后检查原模块入口和类方法已恢复。

成功标记：BUILD_OK、FORWARD_OK、LOSS_OK、LOSS_ALIGNMENT_OK、BACKWARD_OK、
EARLY_EXIT_GATE_GRAD_OK、STEP_OK。LOSS_ALIGNMENT_OK 仅指返回 logits 上的交叉熵计算一致，
不表示模型输出或梯度已与旧版官方参考实现完成数值对齐。

原始失败对照：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh ouro-backward --compat none
```

此命令当前应在 build 阶段因 KeyError: 'default' 失败，退出码为 1。失败也会生成日志和 JSON；
run.sh 的 tee 不会掩盖失败状态。默认适配运行应返回 0。

日志及带模型名、适配模式、时间戳的 JSON 报告保存在 /results/2d1/。
JSON 包含输入、版本、官方模型代码哈希、loss、逐参数梯度范数、各轮输出梯度、
参数更新名单、适配恢复状态及失败时的 traceback。可用 --seed 或 --report 指定随机种子或报告路径。

边界：仅是 tiny 模型的梯度连通性和一步更新检查，尚未验证循环共享参数的参考梯度数值、
完整官方训练目标、真实预训练权重、GPU/BF16、KV cache、分布式或收敛。

## 2D1 收尾：一键完整复跑

一页结论见 [REPORT.md](REPORT.md)，代码落点见 [CODE_PLACEMENT.md](CODE_PLACEMENT.md)。

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh suite
```

顺序执行 10 项 CPU 检查：缓存/tokenizer、五模型导入、LLaDA 前向和反向、
四模型原始/诊断适配前向、两个 LLaDA2 反向、两个 Ouro 反向。
suite 返回 0 表示全部预期得到验证：原始 Ouro 的 KeyError 仍作为已知阻塞被精确复现，
不表示原始 Ouro 在 5.17.0 可以直接使用。意外错误或应通过的项目失败仍返回 1。

每轮生成 /results/2d1/suite-时间-PID/summary.json，以及子报告和 stdout 日志；
run.sh 还记录 Git 分支、HEAD、脚本哈希，并确保 Python assert 不被优化选项关闭。
目前只探测 CPU/FP32/tiny；run.sh 仍隐藏 GPU，不用于未来的 GPU 测试。

单独复跑新增检查：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh assets
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh llada2-backward --model llada21
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh llada2-backward --model llada22
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh ouro-backward --model ouro26
```

assets 默认重算非权重文件 SHA256，核对大权重的清单、文件大小、header 和分片索引，
并实测离线 tokenizer。大权重的完整 SHA256 沿用维护者已有记录；
如需重新扫描约 89 GB 权重，额外加 --full-sha256。

LLaDA2 脚本使用官方同位置 labels 分支，固定一半掩码，只计算被掩码位置的 CE。
检查 train 模式下的 router/共享专家/参与监督的路由专家梯度和更新；
不要求只处理未监督位置的末层专家梯度非零。此检查尚未覆盖正式扩散目标或数值对齐。

GPU 访问只读诊断在 SSH 宿主机执行（不是在容器里执行）：

```bash
bash /mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI/scripts/phase2/2d1/inspect_gpu_access.sh
```

它对比宿主机和容器 nvidia-smi、显示资源限额，并保存日志；任一 GPU 查询失败返回 1。
该环境检查独立于 CPU suite，不会重启容器或修改 GPU 状态。
2026-10-28 实测：宿主机 GPU 正常，容器 NVML 失败；当前 RAM 限额 16 GiB。


## 正式兼容适配层（2026-10-28）

三个模型族现有正式的兼容模型类工厂：
- colossalai/shardformer/modeling/llada.py：get_llada_model_class；
- colossalai/shardformer/modeling/llada2_moe.py：get_llada2_model_class；
- colossalai/shardformer/modeling/ouro.py：get_ouro_model_class。

这些入口接收已从固定快照加载并信任的官方类，返回独立子类；创建后无需保持临时补丁上下文。
LLaDA/LLaDA2 的保存加载元数据已适配 HF5；Ouro 还适配创建和注意力掩码入口。
所有模型复用官方主要计算，配套 native policy 保持参数与模块结构，不执行并行或 kernel 优化。
构造、权重加载和支持边界示例见 [HF5 兼容说明](../../../docs/HF5-COMPATIBILITY.md)。

在开发容器里运行正式适配回归：

```bash
source /home/wzj/venv/bin/activate
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh adapters
```

只看一个模型的相关检查：

```bash
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh adapters -k ouro14
```

check_adapters.py 调用正式 pytest 测试，要求固定快照都在本地，缺少时失败；不会下载大模型。
测试包含五个微型模型的实际 ShardFormer 接入、前向/loss/梯度/更新比较、分片权重往返、
共享权重、Ouro 因果性及快照隔离、未支持配置拒绝，以及已有 D4 policy 回归。
tests/kit/remote_models.py 保存复用的 tiny 配置、输入与构造函数。
日志继续保存至 /results/2d1/，记录本次适配代码和测试文件的 SHA256。

此轮仍是 CPU / FP32 / Transformers 5.17.0 / eager / 无 KV cache。
未验证真实大模型权重加载、GPU/BF16、Booster/ZeRO、多卡、完整扩散训练目标。
原有 suite、remote-tiny 等探测命令继续保留原始模型基线；不要把原始 Ouro 的预期失败误当成适配层回归失败。

验证记录：2026-10-28，105 passed / 0 failed / 0 skipped。
主日志：/results/2d1/adapters-20261028-121109-MVBu0Z.log。
JUnit：/results/2d1/adapters-validation-20261028-1212.xml。
