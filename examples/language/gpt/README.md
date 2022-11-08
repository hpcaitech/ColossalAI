# Run GPT With Colossal-AI

## Overview

In Colossal-AI, there are many ways to run GPT in a distributed manner. The `train_gpt.py` script runs training with the specific configuration scripts in `gpt2_configs/` for different parallelisms of GPT-2 . We have provided some example configuration files of GPT-2 and you can modify them to adapt to your own use.

## How to Prepare Webtext Dataset

We do not host any datasets for GPT or BERT training, however, we provide a detailed guide on how to prepare the dataset so that our results may be reproduced.

### Overview

We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library by [jcpeterson](https://github.com/jcpeterson/openwebtext) and  [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls to different web pages. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in following section.

### Install necessary packages

**Note: LSH requires GCC's early version. We have tested that version 9.3.0 works, but version 10.3.0 is not.**

```bash
pip install ftfy langdetect numpy torch pandas nltk sentencepiece boto3 tqdm regex bs4 newspaper3k htmlmin tldextract cached-path
git clone https://github.com/mattilyra/LSH.git
cd LSH
python setup.py install
```

If you couldn't install it successfully, you may try to replace the `cMinhash.cpp` in `LSH/lsh` with ours, which is provided in `tools/lsh/cMinhash.cpp`.

### Download Data

1. Download the deduplicated URLs from [jcpeterson](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ!cc4RgQQZ).

2. Unzip the zip file and you will get a folder `URLs` which consists of many txt files including urls.

3. Remove blacklisted URLs.

   *We appreciate Megatron-LM for making the data preprocessing code public. We have forked Megatron-LM and fixed some bugs. For your convenience, we have collated the needed files in `tools/Megatron`. Click [here](https://github.com/NVIDIA/Megatron-LM.git) to check the source code of Megatron-LM.*

   ```bash
   cd path/to/tools
   python Megatron/blacklist_urls.py <path/to/URLs> <path/to/clean_urls.txt>
   ```

4. Download the content from the clean urls and merge the contents into one loose json file with 1 json per newline of the format `{'text': text, 'url': unique_url}`.

   *We have forked and modified [openwebtext](https://github.com/yet-another-account/openwebtext) as there are some bugs in it. For your convenience, we provide our modified version in `tools/download`.*

   ```bash
   python download/download.py <path/to/clean_urls.txt> --n_procs 50 --output <path/to/raw.json>
   ```

### Prepare Data for GPT Training

1. Perform ftfy, English detection and remove documents with less than 128 tokens. This step can be sharded and run on shards.

   ```bash
   python Megatron/cleanup_dataset.py <path/to/raw.json> <path/to/clean.json>
   ```

   Additional cleanup (e.g. remove documents less than 512 characters or dataset specific cleaning like stories, realnews datasets) can be done using `cleanup_fix_dataset.py`. More details can be found by running `python cleanup_fix_dataset.py --help`.

2. Using LSH, find possible duplicates and store them in a file for later processing. The code supports saving and loading fingerprints for recurrent deduplications, and is also multithreaded for faster processing. More details are can be found by `python find_duplicate.py --help`.

   ```bash
   python Megatron/find_duplicates.py --inputs <path/to/clean.json> url --output <path/to/process_stage_one.json>
   ```

3. Based on similarity measure defind inside function `is_similar` (default: 0.9), group urls that are similar. Basically, for each group, only one url we should keep and remove the rest.

   ```bash
   python Megatron/group_duplicate_url.py <path/to/process_stage_one.json> <path/to/process_stage_two.json>
   ```

4. Remove similar documents that were detected in the last step. The `dedup.json` is the data after deduplication.

   ```bash
   python Megatron/remove_group_duplicates.py <path/to/process_stage_two.json> <path/to/clean.json> <path/to/dedup.json>
   ```

5. shuffle the dataset.

   ```bash
   shuf <path/to/dedup.json> -o <path/to/train_data.json>
   ```

## How to Prepare Yuan Dataset

### Overview

Yuan dataset is a large scale Chinese dataset with 1TB high quality texts proposed by Inspur. You can apply on https://air.inspur.com/home to get access to the dataset. We downloaded and loaded all downloaded content according to the procedure described in following section.

### Download

The dataset can be according to the website once your application is approved.

You also need to download the vocab file from https://github.com/Shawn-Inspur/Yuan-1.0/blob/main/src/vocab.txt

The final data dir should be organized as:

```
|--dataset
|     |--001.txt
|     |--002.txt
|     |--...
|--vocab.txt
```

### Process & Load

Before you run the code, you should replace line 44 in train_gpt.py with

```
import dataset.yuan import YuanDataset
train_ds = YuanDataset(os.environ['DATA'], vocab_path='/path/to/data/vocab.txt'seq_len=gpc.config.SEQ_LEN)
```

Then you can run model following the Usage section. The dataset will be processed when you run it for the first time, and save the cache. Then the data can be loaded automatically.

## **Usage**

```Bash
#!/usr/bin/env sh
export DATA=/path/to/train_data.json

colossalai run --nproc_per_node=<num_gpus> train_gpt.py --config=gpt2_configs/<config_file>
```

You can copy it and save it as `run.sh`. Then use `bash ./run.sh` to run the script in your terminal.

Please modify `DATA`, `num_gpus` and `config_file` with the path to your dataset, the number of GPUs and the config file path, respectively.
If you are going to train gpt3, just replace `gpt2_configs` with `gpt3_configs`.

## GPT-2

Here are the GPT-2 configs' default parameter:

| config       | scale | GPU* | batch  size | MiB of each GPU | TP  | PP  | DP  |
| ------------ | ----- | ---- | ----------- | --------------- | --- | --- | --- |
| gpt2-vanilla | small | 1    | 1           | 6071            | 1   | 1   | 1   |
| gpt2-vanilla | small | 2    | 1           | 6449*2          | 1   | 1   | 2   |
| gpt2-1d      | small | 2    | 1           | 5287*2          | 2   | 1   | 1   |
| gpt2-2d      | small | 4    | 1           | 4590*4          | 4   | 1   | 1   |
| gpt-2.5d     | small | 8    | 1           | 4815*8          | 8   | 1   | 1   |
| gpt2-3d      | small | 8    | 1           | 4901*8          | 8   | 1   | 1   |
| gpt2-pp      | small | 2    | 1           | 5877*2          | 1   | 2   | 1   |
| gpt2-zero2   | small | 1    | 1           | 5459            | 1   | 1   | 1   |
| gpt2-zero3   | small | 1    | 1           | 6577            | 1   | 1   | 1   |
| gpt2-nvme    | small | 1    | 1           | 5067            | 1   | 1   | 1   |
| gpt2-pp1d    | small | 8    | 8           | 5411*8          | 2   | 2   | 2   |

*\*Note: For GPUs, we use Nvidia A100 80G.*
*\*Note: Results of ZeRO are outdated, we will update them soon.*

**We set** `TENSOR_PARALLEL` `PIPELINE_PARALLEL` **and** `DATA_PARALLEL` **as small as it can be to run every demo with the least number of GPUs.**

### **Modify the config file**

#### **General**

There are some **general rules** when modifying the config files.

```Plain%20Text
TP denotes Tensor Parallel
PP denotes Pipeline Parallel
DP denotes Data Parallel

GPUS = TP * PP * DP
Where DP is autoseted
```

You can set the **batch size** and the **epoch** number by changing the number of
`BATCH_SIZE` and  `NUM_EPOCHS`, respectively. Then, we will introduce the config file of each mode.

Please note that `gpt2_zero3.py` has nothing but `BATCH_SIZE` and `NUM_EPOCHS` to change.

#### **Vanilla & Data Parallel**

`Vanilla` is the basic mode of GPT-2 with no parallelism at all. However, if you use more than 1 GPU and TP * PP < no. of GPUs, Colossal-AI will **set DP for you** **automatically**.

#### **1D, 2D, 2.5D, 3D**

In files `gpt2_1d.py, gpt2_2d.py, gpt2_2p5d.py, gpt2_3d.py`, there is a line:

```Python
TENSOR_PARALLEL = 2
```

You can modify it to use more tensor parallel, just with the general rules satisfied.
In particular, `TENSOR_PARALLEL` should be a square number and cubic number for 2D and 3D,
respectively, and `TENSOR_PARALLEL / DEPTH` should be a square number for 2.5D.

#### **Pipeline Parallel**

To use pipeline parallel training, you should install colossalai from the **latest** main branch.

In `gpt2_pp.py`, there are lines:

```Python
# BATCH_SIZE / NUM_MICRO_BATCHES should be an integer
NUM_MICRO_BATCHES = 1
PIPELINE = 2
```

#### **Pipeline + 1D + Data Parallel**

In `gpt2_pp1d.py`, we have

```Python
BATCH_SIZE = 8
NUM_EPOCHS = 60
NUM_MICRO_BATCHES = 1
HIDDEN_SIZE = 768
PIPELINE = 2
TENSOR_PARALLEL = 2
MODE  = '1d'
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)
```

We have introduced `BATCH_SIZE`, `NUM_EPOCHS`, `NUM_MICRO_BATCHES`, `PIPELINE`, `TENSOR_PARALLEL` as discussed above.
`HIDDEN_SIZE` refers to the hidden dimension of the model, i.e. `gpt2_small` is 768.
You can choose `None, '1d', '2d', '2.5d', '3d'` for `MODE`.

## GPT-3

GPT-3 is a really huge model, for which it seems not possible to train it with a little number of GPUs. Therefore, we choose some common sets of parameters instead of the smallest ones.

Here are our default parameters of GPT-3 configs:

| config         | GPU* | batch size | TP  | PP  | DP  |
| -------------- | ---- | ---------- | --- | --- | --- |
| gpt3_pp1d_min  | 96   | 192        | 4   | 24  | 1   |
| gpt3_pp1d      | 128  | 192        | 4   | 32  | 1   |
| gpt3_pp2d      | 96   | 2*48       | 4   | 24  | 1   |
| gpt3_pp2p5d    | 96   | 2*48       | 4   | 24  | 1   |
| gpt3_zero3_min | 64   | 3          | 1   | 1   | 64  |
| gpt3_zero3     | 96   | 2          | 1   | 1   | 96  |

*\*Note: we use Nvidia A100 40G GPUs*
*\*Note: Results of ZeRO are outdated, we will update them soon.*

In the figure above, the suffix `_min` means the set of hyper-parameters requires the least number of GPUs with the same mode.

GPT-3 and GPT-2 have the same set of hyper-parameters.
