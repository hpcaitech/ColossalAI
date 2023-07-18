# Pretraining
1. Pretraining roberta through running the script below. Detailed parameter descriptions can be found in the arguments.py. `data_path_prefix` is absolute path specifies output of preprocessing. **You have to modify the *hostfile* according to your cluster.**

```bash
bash run_pretrain.sh
```
* `--hostfile`: servers' host name from /etc/hosts
* `--include`: servers which will be used
* `--nproc_per_node`: number of process(GPU) from each server
* `--data_path_prefix`: absolute location of train data, e.g., /h5/0.h5
* `--eval_data_path_prefix`: absolute location of eval data
* `--tokenizer_path`: tokenizer path contains huggingface tokenizer.json, e.g./tokenizer/tokenizer.json
* `--bert_config`: config.json which represent model
* `--mlm`: model type of backbone, bert or deberta_v2

2. if resume training from earlier checkpoint, run the script below.

```shell
bash run_pretrain_resume.sh
```
* `--resume_train`: whether to resume training
* `--load_pretrain_model`: absolute path which contains model checkpoint
* `--load_optimizer_lr`: absolute path which contains optimizer checkpoint
