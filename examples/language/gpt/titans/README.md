# Run GPT With Colossal-AI

## How to Prepare Webtext Dataset

You can download the preprocessed sample dataset for this demo via our [Google Drive sharing link](https://drive.google.com/file/d/1QKI6k-e2gJ7XgS8yIpgPPiMmwiBP_BPE/view?usp=sharing).


You can also avoid dataset preparation by using `--use_dummy_data` during running.

## Run this Demo

Use the following commands to install prerequisites.

```bash
# assuming using cuda 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install colossalai==0.1.9+torch1.11cu11.3 -f https://release.colossalai.org
```

Use the following commands to execute training.

```Bash
#!/usr/bin/env sh
export DATA=/path/to/small-gpt-dataset.json'

# run on a single node
colossalai run --nproc_per_node=<num_gpus> train_gpt.py --config configs/<config_file> --from_torch

# run on multiple nodes with slurm
colossalai run --nproc_per_node=<num_gpus> \
   --master_addr <hostname> \
   --master_port <port-number> \
   --hosts <list-of-hostname-separated-by-comma> \
   train_gpt.py \
   --config configs/<config_file> \
   --from_torch \
   --use_dummy_data

# run on multiple nodes with slurm
srun python \
   train_gpt.py \
   --config configs/<config_file> \
   --host <master_node> \
   --use_dummy_data

```

You can set the `<config_file>` to any file in the `configs` folder. To simply get it running, you can start with `gpt_small_zero3_pp1d.py` on a single node first. You can view the explanations in the config file regarding how to change the parallel setting.
