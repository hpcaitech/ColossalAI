# 添加新的并行技术

为了方便科研人员和工程师们更方便地拓展我们的系统来兼容一些新的大规模分布式训练算法，我们对训练过程中的几个组件进行了解耦，您可以通过继承基类的方式来实现新的并行技术。

主要的组件如下所示：

1. `ProcessGroupInitializer`
2. `GradientHandler`
3. `Schedule`

## 进程组初始化器

并行化一般是通过进程组来进行管理的，同属于一个并行化算法的进程将被分到一个进程组中，如果系统中存在多种不同的并行化技术，那么需要创建多个不同的进程组。Colossal-AI为用户提供了一个全局上下文变量来便捷地管理他们的进程组。如果您希望增加新的进程组，您可以定义一个新的类并且在您的配置文件中进行设置。下方的代码块介绍了如何在系统中加入您的新颖并行技术以及如何进行初始化。

1. 在`colossalai.context.parallel_mode.ParallelMode`中添加新的并行模式。
```python
class ParallelMode(Enum):
    GLOBAL = 'global'
    DATA = 'data'
    PIPELINE = 'pipe'
    PIPELINE_PREV = 'pipe_prev'
    PIPELINE_NEXT = 'pipe_next'
    ...

    NEW_MODE = 'new_mode'  # define your mode here
```

2. 创建一个`ProcessGroupInitializer`的子类，您可以参考`colossalai.context.dist_group_initializer`中给出的例子。前六个参数将由`ParallelContext`决定。如果您需要设置新的参数，您可以用新的参数替换下面例子中的`arg1`与`arg2`。最后，您需要使用`@DIST_GROUP_INITIALIZER.register_module`装饰器在我们的注册表中注册您的初始化器。
```python
# sample initializer class
@DIST_GROUP_INITIALIZER.register_module
class MyParallelInitializer(ProcessGroupInitializer):

    def __init__(self,
                rank: int,
                world_size: int,
                config: Config,
                data_parallel_size: int,
                pipeline_parlalel_size: int,
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

在此之后，您可以将您的初始化器插入到当前的mode-to-initialize映射`colossalai.constants.INITIALIZER_MAPPING`中，您也可以通过更改该文件来动态变更名称与并行模式的映射。

```python
colossalai.constants.INITIALIZER_MAPPING['new_mode'] = 'MyParallelInitializer'
```

3. 在配置文件中设置您的初始化器。如果您的初始化器需要参数，您可以自行传入。下面的代码可以让`ParallelContext`来创建您的初始化器并初始化您需要的进程组。

```python
parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=x, mode='new_mode')  # this is where you enable your new parallel mode
)
```

## 梯度处理器

梯度处理器的功能是对模型参数的梯度进行all-reduce操作。由于不同的并行技术可能需要不同的all-reduce操作，用户们可以通过继承`colossalai.engine.gradient_handler.BaseGradientHandler`来执行其个性化操作。目前，Colossal-AI使用普通的数据并行梯度处理器，该处理器在所有的数据并行rank上执行all-reduce操作，且当Colossal-AI检测到当前系统使用了数据并行时，该处理器会被自动创建。您可以使用下方代码块中的代码添加您自定义的梯度处理器：

```python
from colossalai.registry import GRADIENT_HANDLER
from colossalai.engine import BaseGradientHandler

@GRADIENT_HANDLER.register_module
class YourGradientHandler(BaseGradientHandler):

    def handle_gradient(self):
        do_something()

```

在此之后，您可以在配置文件中指定您想要使用的梯度处理器。

```python
dist_initializer = [
    dict(type='YourGradientHandler'),
]
```

## 调度器

调度器中指定了在前向传播和后向传播时需要执行哪些操作，Colossal-AI提供了流水线和非流水线的调度器。如果您想要修改前向传播和后向传播的执行方式，您可以继承`colossalai.engine.BaseSchedule`并实现您想要的操作。您也可以在训练模型之前将您的调度器添加到我们的引擎中来。
