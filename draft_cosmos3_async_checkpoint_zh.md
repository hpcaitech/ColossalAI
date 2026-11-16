# Cosmos 3 异步 checkpoint：官方实现对比与两条流程的落地

本文对照上游 NVIDIA cosmos-framework 的异步 checkpoint，说明 `examples/worldmodel/cosmos3/train_sft.py`
与 `examples/worldmodel/cosmos3_harness/` 两条流程原来的差距、这次的实现，以及和官方**仍然不完全一致**的地方。
官方结论均核对过源码，文件路径与行号基于 `cosmos_framework/checkpoint/dcp.py`（`NVIDIA/cosmos-framework`）。

## 1. 官方是怎么做的

`DistributedCheckpointer`（`checkpoint/dcp.py:763`）的关键路径只有四步：

1. `save()` 入口先 `_wait_for_previous_async_checkpoint()`（`:1290`）：同一时刻只允许一个未完成的保存，
   上一次没写完就在这里阻塞等待，并把失败统一抛成 `RuntimeError`，保证所有 rank 在同一步退出。
2. 组装 `{model, optim, scheduler, trainer(grad_scaler, iteration, rng_state_<rank>)}`（`:1300-1309`）。
3. `_checkpoint_async_with_pinned_memory()`（`:1064`）：
   - `_create_cpu_state_dict(..., pin_memory=True, share_memory=True)` **只分配一次并在后续保存中复用**（`:1074`）；
   - `staging_stream.wait_stream(current_stream)` 后在独立 CUDA stream 上 `_copy_state_dict(..., non_blocking=True)`（`:1079-1085`）；
   - **`staging_stream.synchronize()`（`:1088`）是唯一暴露给训练步的代价**；
   - `mp_queue_send.put_nowait(...)` 把共享内存张量交给后台进程（`:1091`）。
4. 后台进程（`save_checkpoint_in_background`，`:255`）用预留的 TCPStore socket 自建 **gloo** 进程组
   （`background_store.py`，避免端口被抢占），循环取任务 → `dcp.save(...)`（`CustomSavePlanner(dedup_save_to_lowest_rank=True,
   enable_plan_caching=True)`）→ rank0 写 `latest_checkpoint.txt` → 回 `SaveDone`。

配套语义：

- `poll_async_save()`（`:1112`）每步非阻塞确认成功，让"已落盘"尽早被记账；失败**不在这一步抛**，而是留给
  `_wait_for_previous_async_checkpoint`，因为各 rank 的后台进程完成时间不同，提前抛会让先抛的 rank 掉队。
- `finalize()`（`:1341`）等待未完成保存、发 `Terminate`、join 后台进程。
- 失败即停：`RuntimeError("Previous checkpoint save failed. Exiting...")`（`:1209`）。

一句话：**官方暴露给训练步的成本 = 一次 pinned 暂存拷贝**；序列化、磁盘 I/O、`latest` 指针全在后台，同一时刻
只有一个未完成保存，失败在所有 rank 的同一里程碑炸掉。

## 2. 两条流程原来的状态

| | `cosmos3/train_sft.py`（SFT） | `cosmos3_harness/checkpoint.py`（harness） |
| --- | --- | --- |
| 落盘内容 | 只有模型导出，**没有 optimizer / RNG / sampler** | model + optimizer + 每 rank RNG + `training_state.pt` |
| 目录 | `args.out/step-<N>/`，可覆盖 | `checkpoint-<uuid>/` 不可变代 |
| 完整性 | 无清单、无指针 | SHA256 `complete.json` + 原子 `latest.json` |
| 续训 | **不支持**（README 明说） | `--resume`，集体预检后才写权重 |
| 异步 | `--async-save` 接的是 `use_async=True` | 完全同步 |
| 能不能跑 | **在本项目环境跑不起来** | —— |

两个硬事实：

- ColossalAI 的异步写硬依赖 TensorNVMe（`colossalai/utils/safetensors.py` 的 `_require_async_file_writer`），
  而 TensorNVMe 的 `_C` 扩展是 CUDA 专用。部署环境是 Ascend 910B，`--async-save` 一调用就抛
  `RuntimeError("Async checkpoint I/O requires TensorNVMe")`。`draft_cosmos3_parallel_port_zh.md:223` 也记着它
  "只做了语法检查，一次都没有真跑过"。
- 同步保存实测 47–72 s（`draft_cosmos3_step_time_zh.md`），`save_iter=100` 时平均每步摊 ~0.5 s，而且是一次性
  卡住整个训练循环。

harness 那条虽然内容完整，但 rank0 在关键路径上还要把每个文件重新读一遍算 SHA256，相当于保存后又多了一次
全量读盘。

## 3. 差距表

| 维度 | 官方 | 改造前 |
| --- | --- | --- |
| 暂存 | pinned + 共享内存，分配一次复用；side stream 非阻塞拷贝后一次 synchronize | `pinned_state_dicts` 已复用，但 `copy_` 同步 |
| 后台执行体 | 独立进程 + 自建 gloo 组 | 无（tensornvme pthread writer，NPU 不可用） |
| 节流 | save 入口等上一次 | `save_model()` 入口已有 `_sync_io()`；`save_optimizer()` 没有 |
| 完成确认 | 每步 poll + 阻塞点统一报错 | 无 |
| latest 时机 | 后台写完即更新 | 同步写完更新；异步拖到收尾 |
| 保存内容 | model/optim/scheduler/trainer+rng | harness 齐全；SFT 只有 model |
| 续训 | skip keys、EMA、`dedup_to_lowest_rank`、plan 复用 | harness 有全量 SHA256 预检；SFT 无 |

## 4. 这次改了什么

### 4.1 一个不依赖 TensorNVMe 的后台 writer

`colossalai/utils/safetensors.py` 新增 `ThreadedSafetensorsWriter`，与 TensorNVMe 的 `AsyncFileWriter` 同接口
（`sync_before_step()` / `synchronize()`），`save()` / `move_and_save()` 在 `AsyncFileWriter is None` 时自动回退到它。
它只把序列化和落盘挪到线程：**设备到主机的拷贝仍在 `save()` 返回前完成**（`move_and_save` 的 pinned 路径是同步
`copy_`），所以"训练步只付 D2H"这个性质不变。线程内异常在 `synchronize()` 里重抛，不静默吞掉。

配套修了三处会让它在真机上出问题的地方：

- `_pin_tensor()` 增加一次性能力探测：CPU-only 或某些加速器构建上 `pin_memory=True` 不可用时退化成普通 CPU
  内存，而不是直接抛。
- `_model_sharder` 的 buffer / extra_state 分支原来用 `param_` 去 `empty_like`（`hybrid_parallel_checkpoint_io.py`
  原 187、203 行），形状/类型都可能不对；改用各自的张量。
- `_sync_io()` 原来是"遇到第一个异常就中断"，会把剩下的 writer 留在列表里、staging 缓冲继续存活。改成先排空
  再抛；`__del__` 用不抛的版本。

### 4.2 staging 缓冲的生命周期

pinned 缓冲按 `hash(model)` / `id(optimizer)` 复用，"上一次写还没完就覆写"会静默写坏 checkpoint。
`HybridParallelCheckpointIO` 现在给每个对象记一份**尚未完成的 writer**（`_staging_writers`），在复用该对象的
pinned 缓冲前只 join 这一组（`_join_pending_writers`）。

注意这里**不能**在 `save_optimizer()` 入口无脑加 `_sync_io()`：模型和优化器在同一次保存里是先后入队的，
`_sync_io()` 会把刚入队的模型写立刻 join 掉，异步就没了。所以是按对象精确 join，而不是全量 join。

### 4.3 harness：保存即发布，收尾统一确认

`save_checkpoint(..., async_=True)`：

1. 所有 rank 写自己那份小文件（`rng-<rank>.pt`、rank0 的 `training_state.pt`）并 `barrier()`；
2. `booster.save_model/save_optimizer(use_async=True)` 把分片交给后台 writer；
3. rank0 起**发布线程**：拷贝一份 writer 列表 → 逐个 `synchronize()` → chmod → 逐文件 SHA256 → `complete.json`
   → `os.replace(staging, generation)` → 原子 `latest.json`。

第 3 步不碰任何 `dist.*`，所以它和训练步真正并行；`latest.json` 在写盘完成时立刻更新，与官方"后台进程写完即
写 latest"一致。非保存 rank 根本没有异步 writer，它们的小文件在第 1 步已经落盘并 barrier 过，所以发布线程
不需要再等谁。

`flush_checkpoint(pending)` 是所有 rank 的汇合点：rank0 join 发布线程，各 rank `_sync_io()` 排空自己的 writer，
然后**一次 `all_gather_object`** 同时充当屏障和错误聚合。用一次集合通信而不是"barrier + broadcast"，是为了
不给"某个 rank 在 barrier 之后、broadcast 之前抛异常"留窗口；本地失败也被捕获后照常进入集合通信，不会把
同伴挂在下一个 collective 里。

`complete.json` 的 `format_version` 升到 2（v1 仍可读），RNG 文件新增可选的 `extra` 字段承载 rank 私有随机源。

### 4.4 SFT：统一成可续训代

- 保存改成写 `<out>/checkpoints/checkpoint-<uuid>/`，复用 harness 的 `save_checkpoint`；`--save-every` 的语义
  从"模型导出"变成"可续训代"。
- 新增 `--resume <run dir | checkpoint-*>`；`training_state.pt` 里带一份 `config` 指纹（拓扑、zero stage、
  precision、samples_per_rank、grad_accum、seed、数据根、chunk/fps/resolution、augmentation），不匹配直接拒。
- 游标 `(epoch, sampler_cursor)` 按**微步**计数：一步拉 `grad_accum` 个 batch，epoch 在采样流回绕处前进，
  所有 rank 用同一套算术（`advance_cursor`），所以保存下来的位置与"谁负责拉 batch"无关。
- 续训时 `ResumeDistributedSampler` 直接从本 rank 的索引流里切掉前 `cursor × batch_size` 个索引，
  **不会重新解码**已消费的样本；`set_epoch` 换 epoch 时把偏移清零，同一个 epoch 重设则保留。
- `--resume` 的校验（逐文件 SHA256 + 配置指纹）发生在建 dataloader **之前**，因为 sampler 在构造时就需要
  游标；真正加载权重在 boost 之后、播种之后（`apply_checkpoint`），避免被 `torch.manual_seed` 覆盖。
- rank 私有的 `torch.Generator`（`noise`）通过 `on_rank_state` 随 RNG 文件一起存取。

### 4.5 可观测性

`ckpt/save_ms` 记录保存入口的暴露耗时（异步时就是 staging 的时间），`ckpt/async_pending` 记录未完成保存数；
两者在 `training_metrics.py` 里早已声明聚合方式，此前无人写入。

## 5. 和官方"效果性能一致"到什么程度

**一致的**：

- 暴露给训练步的成本是一次 pinned 暂存（D2H），不是整个序列化 + 落盘 + 哈希。
- 同一时刻只有一个未完成保存：下一次保存入口 join 上一次（`save_model()` 的 `_sync_io()` 加上按对象 join）。
- `latest` 指针在写盘完成即更新，不是等下一次保存。
- 失败在所有 rank 的同一个里程碑一起抛；失败代永远不替换 `latest.json`。
- 保存内容与官方 `CHECKPOINT_KEYS` 对齐：model、optim、每 rank RNG、迭代/step、rank 私有随机源。
  （两条流程都没有 LR scheduler 和 EMA，所以没有这两项。）

**不一致的，别当成完全对齐**：

1. **落盘格式仍是 ColossalAI 的 sharded safetensors，不是 DCP**。好处是两条流程共用一套、现有 SHA256 清单
   与续训校验全部保留、不引入 DCP 依赖；代价是没有官方的 `dedup_to_lowest_rank` 和 save-plan 复用（每次重算
   索引，属于小开销）。
2. **暴露成本里还包含优化器状态的 deepcopy + all-gather + 每 rank 全量 D2H**。
   `_optimizer_sharder` 先 `copy.deepcopy(state)`，ZeRO 下逐参数 `all_gather` 出完整状态，保存 rank 拷进 pinned、
   **非保存 rank 也要 `clone().to("cpu")` 一份全量**。这是"世界大小无关"落盘格式的固有代价；官方按 rank 存分片，
   只拷本地那一片。量级上仍是从 47 s 降到秒级，但不是逐项对齐。要彻底对齐需要新增 shard-local 优化器格式
   （反正续训已经要求同 world size），属后续工作。
3. **执行体是线程，不是独立进程**。官方的独立进程 + gloo 组主要服务于 DCP 的跨 rank 写入计划；我们的分片写入
   本来就只在单个 rank 上发生，`safetensors.save_file` 在 Rust 侧会释放 GIL，剩下的 Python 侧只有索引和哈希
   遍历。真机上仍需实测确认 GIL 影响。
4. **没有官方的每步非阻塞 poll**。完成信号是线程 join，跨 rank 没有共享队列；改为"下一次保存入口 + 收尾"统一
   确认。因为非保存 rank 没有异步写，`latest.json` 的时效不受影响，但**保存失败要等到下一个里程碑才炸**——
   这一点和官方的失败延迟语义相同（官方也是把失败留到阻塞等待点）。
5. **SFT 的在线增广不可逐位复现**。worker 进程有自己的 RNG，续训能复现样本序列、模型、优化器以及每个 rank 的
   Python/NumPy/Torch/加速器状态，但**不能**复现已消费 batch 的逐样本增广抽样。要求完全逐位一致时用
   `--image-augmentation none` 或 `--latent-cache`。harness 的 replay 数据是确定性的，不受此限。

## 6. 怎么验证

CPU（无需加速卡）：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q -o addopts='' tests/test_checkpoint_io/test_safetensors_async_io.py
CUDA_VISIBLE_DEVICES="" python -m pytest -q -o addopts='' examples/worldmodel/cosmos3_harness/tests/test_resume.py
CUDA_VISIBLE_DEVICES="" python -m pytest -q -o addopts='' examples/worldmodel/cosmos3/tests
COLOSSAL_PYTHON=python bash examples/worldmodel/cosmos3_harness/test_ci.sh
```

覆盖点：回退 writer 的往返与异常重抛、`sync_before_step()` 不阻塞而 `synchronize()` 阻塞、`move_and_save` 的
pinned 暂存语义；harness 在 2 rank Gloo 下 `async == sync`（loss/权重/AdamW/RNG 逐位相同）、发布前不可见、
发布失败时全体同错且 `latest.json` 不变；SFT 的保存代格式、游标回绕、sampler 跳过不重解码。

真机（NPU）清单：

1. 单卡 100 step：`--save-every 50 --async-save`，看 `ckpt/save_ms` 从 ~47 s 降到 D2H 量级
   （1.42B 模型 + AdamW 约 14 GB，按 ~25 GB/s 约 0.6 s，实际受 HCCS/PCIe 与 all-gather 影响）。
2. `ckpt/async_pending` 应在保存后 1–2 s 内归零，`latest.json` 在保存后立即出现。
3. 杀掉进程后用 `--resume <run>/checkpoints` 续训，比对中断前后同一 step 的 loss 与 grad norm。
4. 反向验证：把 `--save-every` 设成 1，确认异步与同步两种模式产出的 `training_state.pt` 与权重完全相同。
