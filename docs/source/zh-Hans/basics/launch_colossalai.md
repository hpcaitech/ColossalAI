# 启动 Colossal-AI

作者: Chuanrui Wang, Shenggui Li, Siqi Mai

**预备知识:**
- [分布式训练](../concepts/distributed_training.md)
- [Colossal-AI 总览](../concepts/colossalai_overview.md)


## 简介

正如我们在前面的教程中所提到的，在您的配置文件准备好后，您需要为 Colossal-AI 初始化分布式环境。我们把这个过程称为 `launch`。在本教程中，您将学习如何在您的服务器上启动 Colossal-AI，不管是小型的还是大型的。

在 Colossal-AI 中，我们提供了几种启动方法来初始化分布式后端。
在大多数情况下，您可以使用 `colossalai.launch` 和 `colossalai.get_default_parser` 来通过命令行传递参数。如果您想使用 SLURM、OpenMPI 和 PyTorch 等启动工具，我们也提供了几个启动的辅助方法以便您的使用。您可以直接从这些启动工具设置的环境变量中访问 rank 和 world size 大小。

在本教程中，我们将介绍如何启动 Colossal-AI 来初始化分布式后端：
- 用 colossalai.launch 启动
- 用 Colossal-AI命令行 启动
- 用 SLURM 启动
- 用 OpenMPI 启动

## 启动分布式环境

为了启动 Colossal-AI，我们需要两类参数:
1. 配置文件
2. 分布式设置

无论我们使用何种启动方式，配置文件是必须要求的，而分布式设置有可能依情况而定。配置文件可以是配置文件的路径或 Python dictionary 的形式。分布式设置可以通过命令行或多进程启动器传递。

### 命令行解析器

在使用 `launch` 之前, 我们首先需要了解我们需要哪些参数来进行初始化。
如[分布式训练](../concepts/distributed_training.md) 中 `基本概念` 一节所述 ，涉及的重要参数是:

1. host
2. port
3. rank
4. world_size
5. backend

在 Colossal-AI 中，我们提供了一个命令行解析器，它已经提前添加了这些参数。您可以通过调用 `colossalai.get_default_parser()` 来获得这个解析器。这个解析器通常与 `colossalai.launch` 一起使用。

```python
# add these lines in your train.py
import colossalai

# get default parser
parser = colossalai.get_default_parser()

# if you want to add your own arguments
parser.add_argument(...)

# parse arguments
args = parser.parse_args()
```

您可以在您的终端传入以下这些参数。
```shell

python train.py --host <host> --rank <rank> --world_size <world_size> --port <port> --backend <backend>
```

`backend` 是用户可选的，默认值是 nccl。

### 本地启动

为了初始化分布式环境，我们提供了一个通用的 `colossalai.launch` API。`colossalai.launch` 函数接收上面列出的参数，并在通信网络中创建一个默认的进程组。方便起见，这个函数通常与默认解析器一起使用。

```python
import colossalai

# parse arguments
args = colossalai.get_default_parser().parse_args()

# launch distributed environment
colossalai.launch(rank=args.rank,
                  world_size=args.world_size,
                  host=args.host,
                  port=args.port,
                  backend=args.backend
)

```


### 用 Colossal-AI命令行工具 启动

为了更好地支持单节点以及多节点的训练，我们通过封装PyTorch的启动器实现了一个更加方便的启动器。
PyTorch自带的启动器需要在每个节点上都启动命令才能启动多节点训练，而我们的启动器只需要一次调用即可启动训练。

首先，我们需要在代码里指定我们的启动方式。由于这个启动器是PyTorch启动器的封装，那么我们自然而然应该使用`colossalai.launch_from_torch`。
分布式环境所需的参数，如 rank, world size, host 和 port 都是由 PyTorch 启动器设置的，可以直接从环境变量中读取。

train.py
```python
import colossalai

colossalai.launch_from_torch()
...
```

接下来，我们可以轻松地在终端使用`colossalai run`来启动训练。下面的命令可以在当前机器上启动一个4卡的训练任务。
你可以通过设置`nproc_per_node`来调整使用的GPU的数量，也可以改变`master_port`的参数来选择通信的端口。

```shell
# 在当前节点上启动4卡训练 （默认使用29500端口）
colossalai run --nproc_per_node 4 train.py

# 在当前节点上启动4卡训练，并使用一个不同的端口
colossalai run --nproc_per_node 4 --master_port 29505 test.py
```

如果你在使用一个集群，并且想进行多节点的训练，你需要使用Colossal-AI的命令行工具进行一键启动。我们提供了两种方式来启动多节点任务

- 通过`--hosts`来启动

这个方式适合节点数不多的情况。假设我们有两个节点，分别为`host`和`host2`。我们可以用以下命令进行多节点训练。
比起单节点训练，多节点训练需要手动设置`--master_addr` （在单节点训练中`master_addr`默认为`127.0.0.1`）。同时，你需要确保每个节点都使用同一个ssh port。可以通过--ssh-port设置。

:::caution

多节点训练时，`master_addr`不能为`localhost`或者`127.0.0.1`，它应该是一个节点的**名字或者IP地址**。

:::

```shell
# 在两个节点上训练
colossalai run --nproc_per_node 4 --host host1,host2 --master_addr host1 test.py --ssh-port 22
```


- 通过`--hostfile`来启动

这个方式适用于节点数很大的情况。host file是一个简单的文本文件，这个文件里列出了可以使用的节点的名字。
在一个集群中，可用节点的列表一般由SLURM或者PBS Pro这样的集群资源管理器来提供。比如，在SLURM中，
你可以从`SLURM_NODELIST`这个环境变量中获取到当前分配列表。在PBS Pro中，这个环境变量为`PBS_NODEFILE`。
可以通过`echo $SLURM_NODELIST` 或者 `cat $PBS_NODEFILE` 来尝试一下。如果你没有这样的集群管理器，
那么你可以自己手动写一个这样的文本文件即可。

提供给Colossal-AI的host file需要遵循以下格式，每一行都是一个节点的名字。

```text
host1
host2
```

如果host file准备好了，那么我们就可以用以下命令开始多节点训练了。和使用`--host`一样，你也需要指定一个`master_addr`。
当使用host file时，我们可以使用一些额外的参数：
- `--include`: 设置你想要启动训练的节点。比如，你的host file里有8个节点，但是你只想用其中的6个节点进行训练，
  你可以添加`--include host1,host2,host3,...,host6`，这样训练任务只会在这6个节点上启动。

- `--exclude`: 设置你想排除在训练之外的节点。当你的某一些节点坏掉时，这个参数会比较有用。比如假如host1的GPU有一些问题，无法正常使用，
  那么你就可以使用`--exclude host1`来将其排除在外，这样你就可以训练任务就只会在剩余的节点上启动。

```shell
# 使用hostfile启动
colossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  test.py

# 只使用部分节点进行训练
colossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --include host1 test.py

# 不使用某些节点进行训练
colossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --exclude host2 test.py
```


### 用 SLURM 启动

如果您是在一个由 SLURM 调度器管理的系统上， 您也可以使用 `srun` 启动器来启动您的 Colossal-AI 脚本。我们提供了辅助函数 `launch_from_slurm` 来与 SLURM 调度器兼容。
`launch_from_slurm` 会自动从环境变量 `SLURM_PROCID` 和 `SLURM_NPROCS` 中分别读取 rank 和 world size ，并使用它们来启动分布式后端。

您可以在您的训练脚本中尝试以下操作。

```python
import colossalai

colossalai.launch_from_slurm(
    host=args.host,
    port=args.port
)
```

您可以通过在终端使用这个命令来初始化分布式环境。

```bash
srun python train.py --host <master_node> --port 29500
```

### 用 OpenMPI 启动
如果您对OpenMPI比较熟悉，您也可以使用 `launch_from_openmpi` 。
`launch_from_openmpi` 会自动从环境变量
`OMPI_COMM_WORLD_LOCAL_RANK`， `MPI_COMM_WORLD_RANK` 和 `OMPI_COMM_WORLD_SIZE` 中分别读取local rank、global rank 和 world size，并利用它们来启动分布式后端。

您可以在您的训练脚本中尝试以下操作。
```python
colossalai.launch_from_openmpi(
    host=args.host,
    port=args.port
)
```

以下是用 OpenMPI 启动多个进程的示例命令。
```bash
mpirun --hostfile <my_hostfile> -np <num_process> python train.py --host <node name or ip> --port 29500
```

- --hostfile: 指定一个要运行的主机列表。
- --np: 设置总共要启动的进程（GPU）的数量。例如，如果 --np 4，4个 python 进程将被初始化以运行 train.py。

<!-- doc-test-command: echo  -->
