# Add your own parallelism

## Overview

To enable researchers and engineers to extend our system to other novel large-scale distributed training algorithm
with less effort, we have decoupled various components in the training lifecycle. You can implement your own
parallelism by simply inheriting from the base class.

The main components are:

1. `ProcessGroupInitializer`
2. `GradientHandler`
3. `Schedule`

## Process Group Initializer

Parallelism is often managed by process groups where processes involved in the same parallel algorithm are placed in the same
process group. For different parallel algorithms, different process groups need to be created. Colossal-AI provides a
global context for users to easily manage their process groups. If you wish to add new process group, you can easily
define a new class and set it in your configuration file. To define your own way of creating process groups, you can
follow the steps below to create a new distributed initialization.

1. Add your parallel mode in `colossalai.context.parallel_mode.ParallelMode`.
    ```python
    class ParallelMode(Enum):
        GLOBAL = 'global'
        DATA = 'data'
        PIPELINE = 'pipe'
        ...

        NEW_MODE = 'new_mode'  # define your mode here
    ```

2. Create a `ProcessGroupInitializer`. You can refer to examples given in `colossalai.context.dist_group_initializer`. The
   first six arguments are fixed. `ParallelContext` will pass in these arguments for you. If you need to set other
   arguments, you can add it behind like the `arg1, arg2` in the example below. Lastly, register your initializer to the
   registry by adding the decorator `@DIST_GROUP_INITIALIZER.register_module`.
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

    Then, you can insert your new initializer to the current mode-to-initialize mapping
    in `colossalai.constants.INITIALIZER_MAPPING`. You can modify the file or insert new key-value pair dynamically.

    ```python
    colossalai.constants.INITIALIZER_MAPPING['new_mode'] = 'MyParallelInitializer'
    ```

3. Set your initializer in your config file. You can pass in your own arguments if there is any. This allows
   the `ParallelContext` to create your initializer and initialize your desired process groups.

    ```python
    parallel = dict(
        pipeline=dict(size=1),
        tensor=dict(size=x, mode='new_mode')  # this is where you enable your new parallel mode
    )
    ```

## Gradient Handler

Gradient handlers are objects which execute the all-reduce operations on parameters' gradients. As different all-reduce
strategies may be executed for different kinds of parallelism, users can
inherit `colossalai.engine.gradient_handler.BaseGradientHandler` to implement their strategies. Currently, the library
uses the normal data parallel gradient handler which all-reduces the gradients across data parallel ranks. The data
parallel gradient handler is added to the engine automatically if data parallel is detected. You can add your own
gradient handler like below:

```python
from colossalai.registry import GRADIENT_HANDLER
from colossalai.engine import BaseGradientHandler

@GRADIENT_HANDLER.register_module
class YourGradientHandler(BaseGradientHandler):

    def handle_gradient(self):
        do_something()

```

Afterwards, you can specify the gradient handler you want to use in your configuration file.

```python
gradient_handlers = [
    dict(type='YourGradientHandler'),
]
```

## Schedule

Schedule entails how to execute a forward and backward pass. Currently, Colossal-AI provides pipeline and non-pipeline
schedules. If you want to modify how the forward and backward passes are executed, you can
inherit `colossalai.engine.schedule.BaseSchedule` and implement the `forward_back_step` function.