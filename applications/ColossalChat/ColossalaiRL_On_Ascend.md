# ColossalaiRL On Ascend
The document is the instructions for using ColossalRL on Ascend.

## 1.Prepare Develop Environment

### Install Colossalai & ColossalChat
```bash
git clone https://github.com/hpcaitech/ColossalAI.git
git checkout grpo-latest
pip install -e .

cd ./applications/ColossalChat
pip install -e .
```

### Install Fuyao Ray
Please update CANN before install fuyao ray
```bash
# Install CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./Ascend-cann-kernels-910b_8.1.RC1.alpha001_linux-aarch64.run  --devel

# Clone Fuyao Ray
git clone https://gitee.com/openfuyao/ray.git
cd ray
git pull origin pull/5/head

# Install ray
pip install ray==2.43.0 --no-cache-dir

# Create soft-link from fuyao-ray to ray site-package
cd ..
ln -s ./ray/python/ray/ /usr/local/python3.10/lib/python3.10/site-packages/ray 

# Install Fuyao Ray
cd ray
python python/ray/setup-dev.py
```
### Prepare Model & dataset

```bash
huggingface-cli download --local-dir-use-symlinks False Qwen/Qwen2.5-7B --local-dir /models/Qwen/Qwen2.5-7B
```


## 2.Set Distributed Config
Now, we need to set distributed config for multi-node.

### Set Host IP Config
First, we set host ip config.
For example. I need to configure a cluster of 4 nodes, then I do
```bash
vim /etc/hosts
```
Then write IP node map to /etc/hosts
```bash
10.0.0.3 npu-3
10.0.0.4 npu-4
10.0.0.5 npu-5
10.0.0.6 npu-6
```

### Set Ascend Multi-Node Config 

```bash
export ATB_LLM_HCCL_ENABLE=1
export ATB_LLM_COMM_BACKEND="hccl"
export HCCL_CONNECT_TIMEOUT=7200
export WORLD_SIZE=32
export HCCL_EXEC_TIMEOUT=7200 
export HCCL_SOCKET_IFNAME=eno0
export RAY_COLLECTIVE_MEET_TIMEOUT_SECONDS=7200 
```

## 3.Run task on ColossalaiRL-Ascend 

### Start Ray Cluster
Now we use 10.0.0.3 as master node. First we start a ray cluster on 10.0.0.3:
```bash
ray start --head --node-ip-address=10.0.0.3
```
Then, for each slave node (10.0.0.4/10.0.0.5/10.0.0.6), we add to the ray cluser by following code:
```bash
ray start --address='10.0.0.3:6379'
```

### Run Scripts
Then, run start command at master node
```bash
# Hint1: replace /models/Qwen/Qwen2.5-7B to your model path
#        replace /datasets/train-alignment.jsonl to your dataset path
python rl_example.py -m /models/Qwen/Qwen2.5-7B -d /datasets/train-alignment.jsonl --master_address '10.0.0.3' -t 16 -i 16 -p GRPO-Train-Align-Debug -g 2 -ibs 1 -tbs 2 -tMbs 1  -tmbs 2 -imbs 1 -b vllm -e 2 -rt boxed -s "Please reason step by step, and put your final answer within \\boxed{}." &>run_log.log &
```

<!-- doc-test-command: echo  -->
