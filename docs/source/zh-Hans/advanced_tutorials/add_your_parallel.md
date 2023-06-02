# 添加你自己的并行模式

作者: Shenggui Li, Yongbin Li

**前置教程**
- [定义配置文件](../basics/define_your_config.md)
- [并行配置](../basics/configure_parallelization.md)

## 引言

为了使研究人员和工程师能够以更少的努力将我们的系统扩展到其他新颖的大规模分布式训练算法，我们已经将训练生命周期中的各种组件解耦。你可以通过简单地继承基类来实现你自己的并行模式。

主要组件有:

1. `ProcessGroupInitializer`
2. `GradientHandler`
3. `Schedule`

**目前这需要对源代码进行一些改动，因此我们建议你用`-e`标志从源代码安装。`-e`标志使得安装是可编辑的，因此，你的代码变化将反映在你的Python运行时中。我们将在这方面努力，以避免在未来的版本中改变源代码。**


## 进程组初始化器

并行通常由进程组来管理，参与相同并行算法的进程被置于同一进程组。对于不同的并行算法，需要创建不同的进程组。
Colossal-AI 为用户提供了一个全局 context，使他们能够轻松地管理进程组。如果你想添加新的进程组，你可以很容易地定义一个新的类并在你的配置文件中设置它。为了定义你自己的进程组创建方式，你可以按照下面的步骤来创建一个新的分布式初始化。

1. 在 `colossalai.context.parallel_mode.ParallelMode` 中添加你自己的并行模式。
    ```python
    class ParallelMode(Enum):
        GLOBAL = 'global'
        DATA = 'data'
        PIPELINE = 'pipe'
        ...

        NEW_MODE = 'new_mode'  # define your mode here
    ```

2. 创建一个 `ProcessGroupInitializer`。 你可以参考 `colossalai.context.dist_group_initializer` 中给出的例子，前六个参数是固定的。
`ParallelContext` 将为你传入这些参数。如果你需要设置其他参数，可以像下面的例子中的 `arg1, arg2` 一样，在后面添加它。
最后，通过添加装饰器 `@DIST_GROUP_INITIALIZER.register_module` 将你的初始化程序注册到注册表。
    ```python
    # sample initializer class
    @DIST_GROUP_INITIALIZER.register_module
    class MyParallelInitializer(ProcessGroupInitializer):

        def __init__(self,
                    rank: int,
                    world_size: int,
                    config: Config,
                    data_parallel_size: int,
                    pipeline_parallel_size: int,
                    tensor_parallel_size: int,
                    arg1,
                    arg2):
            super().__init__(rank, world_size, config)
            self.arg1 = arg1
            self.arg2 = arg2
            # ... your variable init

        def init_parallel_groups(self):
            # initialize your process groups
            pass

    ```
    然后，你可以将你的新初始化器插入到 `colossalai.constants.INITIALIZER_MAPPING` 当前的模式与初始化映射中。你可以修改该文件或动态插入新的键值对。

    ```python
    colossalai.constants.INITIALIZER_MAPPING['new_mode'] = 'MyParallelInitializer'
    ```

3. 在你的配置文件中设置你的初始化器。你可以传入你的自定义参数。这允许
   `ParallelContext` 创建你的初始化器并初始化你期望的进程组。

    ```python
    parallel = dict(
        pipeline=dict(size=1),
        tensor=dict(size=x, mode='new_mode')  # this is where you enable your new parallel mode
    )
    ```

## 梯度 Handler

梯度 handler 是对参数的梯度执行 all-reduce 操作的对象。由于不同的 all-reduce 策略或许在不同的并行中被执行，用户可以继承
`colossalai.engine.gradient_handler.BaseGradientHandler` 来实现其策略。目前，Colossal-AI 使用普通的数据并行梯度 handler 在数据并行的 rank 间 all-reduce 梯度。
如果数据并行被检测到，梯度 handler 会被自动添加进 engine。

你可以添加你自己的梯度 handler，如下所示：

```python
from colossalai.registry import GRADIENT_HANDLER
from colossalai.engine import BaseGradientHandler

@GRADIENT_HANDLER.register_module
class YourGradientHandler(BaseGradientHandler):

    def handle_gradient(self):
        do_something()

```

之后，你可以在配置文件中指定你要使用的梯度 handler。

```python
gradient_handlers = [
    dict(type='YourGradientHandler'),
]
```

## Schedule

Schedule 包含了如何执行前向和后向计算。目前， Colossal-AI 提供了流水和非流水的 schedule。
如果你想修改前向和后向计算的执行方式，你可以继承 `colossalai.engine.schedule.BaseSchedule` 并实现 `forward_back_step` 函数。
