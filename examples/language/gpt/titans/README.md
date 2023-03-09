# Run GPT With Colossal-AI

## How to Prepare Webtext Dataset

You can download the preprocessed sample dataset for this demo via our [Google Drive sharing link](https://drive.google.com/file/d/1QKI6k-e2gJ7XgS8yIpgPPiMmwiBP_BPE/view?usp=sharing).


You can also avoid dataset preparation by using `--use_dummy_dataset` during running.

## Run this Demo

Use the following commands to install prerequisites.

```bash
# assuming using cuda 11.3
pip install -r requirements.txt
```

Use the following commands to execute training.

```Bash
#!/usr/bin/env sh
# if you want to use real dataset, then remove --use_dummy_dataset
# export DATA=/path/to/small-gpt-dataset.json'

# run on a single node
colossalai run --nproc_per_node=<num_gpus> train_gpt.py --config configs/<config_file> --from_torch --use_dummy_dataset

# run on multiple nodes
colossalai run --nproc_per_node=<num_gpus> \
   --master_addr <hostname> \
   --master_port <port-number> \
   --hosts <list-of-hostname-separated-by-comma> \
   train_gpt.py \
   --config configs/<config_file> \
   --from_torch \
   --use_dummy_dataset

# run on multiple nodes with slurm
srun python \
   train_gpt.py \
   --config configs/<config_file> \
   --host <master_node> \
   --use_dummy_dataset

```

You can set the `<config_file>` to any file in the `configs` folder. To simply get it running, you can start with `gpt_small_zero3_pp1d.py` on a single node first. You can view the explanations in the config file regarding how to change the parallel setting.
