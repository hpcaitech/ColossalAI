# Launch Colossal-AI

Author: Chuanrui Wang, Shenggui Li, Siqi Mai

**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)


## Introduction

As mentioned in the previous tutorials stated in the prerequisite, you need to initialize the distributed environment
for Colossal-AI after your config file is prepared.
We call this process `launch`.
In this tutorial, you will learn how to launch Colossal-AI on your server, be it a small one or big one.

In Colossal-AI, we provided several launch methods to initialize the distributed backend.
In most cases, you can use `colossalai.launch` and `colossalai.get_default_parser` to pass the
parameters via command line.
If you happen to use launchers such as SLURM, OpenMPI and PyTorch launch utility,
we also provide several launching helper methods to access the rank and world size from the environment variables
set by these launchers directly for your convenience.

In this tutorial we will cover how to launch Colossal-AI to initialize the distributed backends:
- Launch with `colossalai.launch`
- Launch with Colossal-AI CLI
- Launch with SLURM
- Launch with OpenMPI

## Launch Distributed Environment

In order to launch Colossal-AI, we need two types of arguments:
1. config file
2. distributed settings

The config file is always required regardless of the launch method but distributed settings can vary. The config file
can be a path to the configuration file or a Python dictionary. The distributed settings can be passed via command line
or multi-process launchers.

### Command Line Parser

Before we jump to `launch`, we firstly need to understand what parameters we need for initialization.
As stated in the `Basic Concepts in Distributed Training` section of [Distributed Training](../concepts/distributed_training.md),
the important parameters are:

1. host
2. port
3. rank
4. world_size
5. backend

In Colossal-AI, we provided a command line parser which has added these arguments in advance. You can get this parser by calling
`colossalai.get_default_parser()`. This parser is usually used with `colossalai.launch`.

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

Then in your terminal, you can pass in these arguments:
```shell

python train.py --host <host> --rank <rank> --world_size <world_size> --port <port> --backend <backend>
```

`backend` is optional and the default value is `nccl`.

### Native Launch

To initialize the distributed environment, we provided a general `colossalai.launch` API. The `colossalai.launch` function takes in the parameters
listed above and create a default process group in the communication network. This function is often used with the default
parser for convenience.

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


### Launch with Colossal-AI CLI

To enable easy launching on both single or multi nodes, we have implemented a launcher for Colossal-AI. This launcher is
a wrapper of the torch distributed launch utility but enhanced with the capability of launching multi-node jobs easily.

First, we need to set the launch method in our code. As this is a wrapper of the torch distributed launch utility, we will
use `colossalai.launch_from_torch`. The arguments required for distributed environment such as rank, world size, host and port are all set by the PyTorch
launcher and can be read from the environment variable directly.

train.py
```python
import colossalai

colossalai.launch_from_torch()
...
```

Next, we can easily start multiple processes with `colossalai run` in your terminal. Below is an example to run the code
on a single node with 4 GPUs. You can change the number of GPUs by `nproc_per_node` and the default port by `master_port`.

```shell
# run on the local node with 4 GPUs (default port: 29500)
colossalai run --nproc_per_node 4 train.py

# run on the local node with 4 GPUs with a different port
colossalai run --nproc_per_node 4 --master_port 29505 test.py
```

If you are in a cluster and want to launch multi-node training, the CLI can help you start processes on different nodes
with one simple command. There are two ways you can launch multi-node jobs.

- Run with `--hosts`

This is suitable when you only have a few nodes. Let's say I have two nodes, namely `host1` and `host2`,  I can start
multi-node training with the following command. Compared to single-node training, you must specify the `master_addr`
option, which is auto-set to localhost if running on a single node only. \
Additionally, you must also ensure that all nodes share the same open ssh port, which can be specified using --ssh-port.

:::caution

`master_addr` cannot be localhost when running on multiple nodes, it should be the **hostname or IP address** of a node.

:::

```shell
# run on these two nodes
colossalai run --nproc_per_node 4 --host host1,host2 --master_addr host1 test.py --ssh-port 22
```
- Run with `--hostfile`

This method is suitable when you have a lot of nodes. The host file is a simple text file listing the available nodes.
The list of nodes is commonly provided by cluster managers such as SLURM and PBS Pro. For example, you can get the list
of nodes allocated to you via the environment variable `SLURM_NODELIST` in SLURM and `PBS_NODEFILE` in PBS Pro.
Just do `echo $SLURM_NODELIST` or `cat $PBS_NODEFILE` to check it out. If you do not have such cluster managers, you can
manually create one for your own use.

The host file given to Colossal-AI launcher must be in the following format where each line is the host name of a node.

```text
host1
host2
```

With the host file ready, we can launch multi-node jobs with the following commands. Just like using `--host`, you also
need to specify the `master_addr` option. Some extra options are provided for `--hostfile` as listed below:

- `--include`: specify the hosts to include for multi-node jobs. For example, if your host file has 8 nodes, but you
happen to only want to run on 6 nodes instead, you can add `--include host1,host2,host3,...,host6` so that the job will only
be launcher on the 6 nodes.
- `--exclude`: specify the hosts to exclude for multi-node jobs. This is useful when some nodes are faulty. For example,
if host1 GPU has some problems and you do not wish to run on host1 but all other nodes, you can add `--exclude host1` so that
the job will only be launched on the remaining nodes.

```shell
# run with a hostfile
colossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  test.py

# only include certain hosts to execute commands
# this is used to manually select nodes to run
colossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --include host1 test.py

# exclude certain hosts to execute commands
# this can be used when certain nodes are faulty
colossalai run --nproc_per_node 4 --hostfile ./hostfile --master_addr host1  --exclude host2 test.py
```

### Launch with SLURM

If you are on a system managed by the SLURM scheduler, you can also rely on the `srun` launcher to kickstart your Colossal-AI scripts.
We provided the helper function `launch_from_slurm` for compatibility with the SLURM scheduler.
`launch_from_slurm` will automatically read the rank and world size from the environment variables `SLURM_PROCID` and `SLURM_NPROCS` respectively
and use them to start the distributed backend.
Do this in your training script:

```python
import colossalai

colossalai.launch_from_slurm(
    host=args.host,
    port=args.port
)
```

You can initialize the distributed environment by using this command in terminal.

```bash
srun python train.py --host <master_node> --port 29500
```

### Launch with OpenMPI
If you are more familiar with OpenMPI, you can use `launch_from_openmpi` instead.
`launch_from_openmpi` will automatically read the local rank, global rank and world size from the environment variables
`OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_COMM_WORLD_RANK` and `OMPI_COMM_WORLD_SIZE` respectively and
use them to start the distributed backend.

Do this in your train.py:
```python
colossalai.launch_from_openmpi(
    host=args.host,
    port=args.port
)
```

A sample command to launch multiple processes with OpenMPI would be:

```bash
mpirun --hostfile <my_hostfile> -np <num_process> python train.py --host <node name or ip> --port 29500
```

- --hostfile: use this option to specify a list of hosts on which to run
- --np: set the number of processes (GPUs) to launch in total. For example, if --np 4, 4 python processes will be initialized to run train.py.

<!-- doc-test-command: echo  -->
