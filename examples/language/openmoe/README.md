## OpenMoE
[OpenMoE](https://github.com/XueFuzhao/OpenMoE) is the open-source community's first decoder-only MoE transformer. OpenMoE is implemented in Jax, and [Colossal-AI](https://github.com/hpcaitech/ColossalAI) has pioneered an efficient open-source support for this model in PyTorch, enabling a broader range of users to participate in and use this model. The following example of [Colossal-AI](https://github.com/hpcaitech/ColossalAI) demonstrates finetune and inference methods.


<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/MOE_training.png" width=800/>
</p>

* [2023/11] [Enhanced MoE Parallelism, Open-source MoE Model Training Can Be 9 Times More Efficient](https://www.hpc-ai.tech/blog/enhanced-moe-parallelism-open-source-moe-model-training-can-be-9-times-more-efficient)
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/openmoe)
[[blog]](https://www.hpc-ai.tech/blog/enhanced-moe-parallelism-open-source-moe-model-training-can-be-9-times-more-efficient)

## Usage

### 1. Installation

Please install the latest ColossalAI from source.

```bash
CUDA_EXT=1 pip install -U git+https://github.com/hpcaitech/ColossalAI
```

Then install dependencies.

```bash
cd ColossalAI/examples/language/openmoe
pip install -r requirements.txt
```

Additionally, we recommend you to use torch 1.13.1. We've tested our code on torch 1.13.1 and found it's compatible with our code and flash attention.

### 2. Install kernels (Optional)

We have utilized `Triton`, `FlashAttention` and `Apex` kernel for better performance. They are not necessary but we recommend you to install them to fully utilize your hardware.
```
# install triton via pip
pip install triton

# install flash attention via pip
pip install flash-attn==2.0.5

# install apex from source
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 741bdf50825a97664db08574981962d66436d16a
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ --global-option="--cuda_ext"
```
### 3. Inference
You can inference by the following code to try OpenMoE-8B-Chat model:
```
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model_path = "OrionZheng/openmoe-8b-chat"
config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True, 
    device_map='auto'
    )
query = 'Question: How do I kill a process? Answer:'
prompt = f'''<<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

<s>[INST] {query} [/INST]'''

inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
sample = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(sample[0]))
```

We also provide a Colab [tutorial](https://colab.research.google.com/drive/1eIT1rtG7pORRQAYtQoMOAekUg7aZLDdn) demonstrating the jax checkpoint conversion and execution of PyTorch model inference. You can experiment with OpenMoE-8B-Chat on Colab directly by [this](https://colab.research.google.com/drive/1xIfIVafnlCP2XVICmRwkUFK3cwTJYjCY)(Note: both require Colab Pro).
- Running OpenMoE-8B requires ~49GB of memory in float32 or ~23GB in bfloat16. It can be executed on a Colab `CPU High-RAM` runtime or an `A100-40GB` runtime, both of which require Colab Pro.The float16 precision is not recommended because sometimes it will lead to performance degradation.
- Runing the OpenMoE-34B requries ~89GB of memory in bfloat16 or ~180GB in float32. To perform inference on multiple devices/offloading model weights to RAM, please refer to the script [here](inference_on_multi_devices.py).

### 4. Train
Yon can use colossalai run to launch single-node training:
```bash
colossalai run --standalone --nproc_per_node YOUR_GPU_PER_NODE train.py --OTHER_CONFIGURATIONS
```
Yon can also use colossalai run to launch multi-nodes training:
```bash
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE train.py --OTHER_CONFIGURATIONS
```

Here is a sample hostfile:

```text
hostname1
hostname2
hostname3
hostname4
```

The hostname refers to the ip address of your nodes. Make sure master node can access all nodes (including itself) by ssh without password.

Here is details about CLI arguments:

- Model configuration: `--model_name`. `base` and `8b` are supported for OpenMoE.
- Booster plugin: `--plugin`. `ep`, `ep_zero` and `hybrid` are supported. `ep_zero` is recommended for general cases. `ep` can provides least memory consumption and `hybrid` suits large scale training.
- Output path: `--output_path`. The path to save your model. The default value is `./outputs`.
- Number of epochs: `--num_epochs`. The default value is 1.
- Local batch size: `--batch_size`. Batch size per GPU. The default value is 1.
- Save interval: `-i`, `--save_interval`. The interval (steps) of saving checkpoints. The default value is 1000.
- Mixed precision: `--precision`. The default value is "bf16". "fp16", "bf16" and "fp32" are supported.
- Max length: `--max_length`. Max sequence length. Default to 2048.
- Dataset: `-d`, `--dataset`. The default dataset is `yizhongw/self_instruct`. It support any dataset from `datasets` with the same data format as it.
- Task Name: `--task_name`. Task of corresponding dataset. Default to `super_natural_instructions`.
- Learning rate: `--lr`. The default value is 1e-5.
- Weight decay: `--weight_decay`. The default value is 0.
- Zero stage: `--zero_stage`. Zero stage. Recommend 2 for ep and 1 for ep zero.
- Extra dp size: `--extra_dp_size`. Extra moe param dp size for ep_zero plugin. Recommended to be 2 or 4.
- Use kernel: `--use_kernel`. Use kernel optim. Need to install flash attention and triton to enable all kernel optimizations. Skip if not installed.
- Use layernorm kernel: `--use_layernorm_kernel`. Use layernorm kernel. Need to install apex. Raise error if not installed.
- Router aux loss factor: `--router_aux_loss_factor`. Moe router z loss factor. You can refer to STMoE for details.
- Router z loss factor: `--router_z_loss_factor`. Moe router aux loss factor. You can refer to STMoE for details.
- Label smoothing: `--label_smoothing`. Label smoothing.
- Z loss factor: `--z_loss_factor`. The final outputs' classification z loss factor.
Load balance: `--load_balance`. Expert load balance. Defaults to False. Recommend enabling.
- Load balance interval: `--load_balance_interval`. Expert load balance interval.
- Communication overlap: `--comm_overlap`. Use communication overlap for MoE. Recommended to enable for multi-node training.

### 4. Shell Script Examples

For your convenience, we provide some shell scripts to train with various configurations. Here we will show an example of how to run training
OpenMoE.

#### a. Running environment
This experiment was performed on a single computing nodes with 8 A800 80GB GPUs in total for OpenMoE-8B. The GPUs are fully connected with NVLink.

#### b. Running command
We demonstrate how to run three plugins in `train.sh`. You can choose anyone and use your own args.

```bash
bash train.sh
```

#### c. Multi-Nodes Training

To run on multi-nodes, you can modify the script as:
```bash
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE \
train.py --OTHER_CONFIGURATIONS
```

## Reference
```
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```

```bibtex
@misc{openmoe2023,
  author = {Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou and Yang You},
  title = {OpenMoE: Open Mixture-of-Experts Language Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XueFuzhao/OpenMoE}},
}
```
