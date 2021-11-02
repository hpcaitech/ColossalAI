# ZeRO优化器与offload

ZeRO优化器可以切分三种模型状态（优化器状态、梯度、参数），并将它们存储在不同的进程中，以此来减少数据并行的存储冗余，传统的数据并行需要将上述三种状态
复制很多份保存在每一个进程中。与传统的做法相比，ZeRO优化器可以极大地提高内存存储效率，并保持较好的通信效率。

1. **ZeRO Level 1**: 优化器状态（如对于[Adam优化器](https://arxiv.org/abs/1412.6980)而言，32比特的参数，以及第一和第二动量的预测值）被切分
存储在不同的进程中，这样每一个进程只需要更新它对应的那一部分参数。
2. **ZeRO Level 2**: 用于更新模型参数的32比特的梯度在这一级被切分存储在不同的进程中，这里梯度的切分与level 1中模型参数的切分是一一对应的，每一个
进程上的梯度正好被用来更新该进程上的保存的模型参数。
3. **ZeRO Level 3**: 16比特的模型参数在这一级被切分存储在不同的进程中，ZeRO-3可以在前向传播和后向传播期间自动收集或切分这些参数。

## 使用ZeRO优化器

在ColossalAI中启用ZeRO优化器只需要您在配置文件中进行配置即可，下面是一些使用ZeRO-3的配置文件例子。

### 使用ZeRO优化器以及offload

这里我们使用`Adam`作为我们的初始优化器.

1. 使用ZeRO来切分优化器状态（level 1），梯度（level 2），以及模型参数（level 3）：
    ```python
    optimizer = dict(
        type='Adam',
        lr=0.001,
        weight_decay=0
    )

    zero = dict(
        type='ZeroRedundancyOptimizer_Level_3',
        dynamic_loss_scale=True,
        clip_grad=1.0
    )
    ```
2. 将优化器状态以及计算分配到CPU上：
    ```python
    zero = dict(
        offload_optimizer_config=dict(
            device='cpu',
            pin_memory=True,
            fast_init=True
        ),
        ...
    )
    ```
3. 将模型参数分配到CPU上来节省显存：
    ```python
    zero = dict(
        offload_optimizer_config=dict(
            device='cpu',
            pin_memory=True,
            fast_init=True
        ),
        offload_param_config=dict(
            device='cpu',
            pin_memory=True,
            fast_init=OFFLOAD_PARAM_MAX_IN_CPU
        ),
        ...
    )
    ```
4. 将参数分配到NVMe上来节省更多显存（如果您的系统上安装了NVMe）：
    ```python
    zero = dict(
        offload_optimizer_config=dict(
            device='nvme',
            pin_memory=True,
            fast_init=True,
            nvme_path='/nvme_data'
        ),
        offload_param_config=dict(
            device='nvme',
            pin_memory=True,
            max_in_cpu=OFFLOAD_PARAM_MAX_IN_CPU,
            nvme_path='/nvme_data'
        ),
        ...
    )
    ```

请注意使用ZeRO时`fp16`将会被自动激活。

### 使用ZeRO优化器进行训练

如果您完成了上述配置，可以运行`colossalai.initialize()`来开始您的训练。
