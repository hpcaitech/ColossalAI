<div align="center">
<h1>
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/colossaleval.jpg?raw=true" width=800/>
</h1>

 <h3>
 <a href="https://cloud.luchentech.com/">GPU Cloud Playground </a> </a> |
 <a href="https://cloud.luchentech.com/doc/docs/image/colossal-eval"> Colossal-Eval Image </a>
 </h3>

</div>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Leaderboard](#leaderboard)
  - [Model with ~13 Billion Parameters](#model-with-13-billion-parameters)
  - [Model with ~7 Billion Parameters](#model-with-7-billion-parameters)
- [Install](#install)
- [Evaluation Process](#evaluation-process)
  - [Inference](#inference)
    - [Dataset Preparation](#dataset-preparation)
    - [Configuration](#configuration)
    - [How to Use](#how-to-use)
  - [Evaluation](#evaluation)
    - [Dataset Evaluation](#dataset-evaluation)
      - [Configuration](#configuration-1)
      - [How to Use](#how-to-use-1)
    - [GPT Evaluation](#gpt-evaluation)
      - [Configuration](#configuration-2)
      - [How to Use](#how-to-use-2)
- [More Details](#more-details)
  - [Inference](#inference-1)
  - [Evaluation](#evaluation-1)
    - [Metrics](#metrics)
  - [Examples](#examples)
    - [Dataset Evaluation Example](#dataset-evaluation-example)
    - [GPT Evaluation Example](#gpt-evaluation-example)
- [FAQ](#faq)
  - [How to Add a New Metric?](#how-to-add-a-new-metric)
  - [How to Add a New Dataset?](#how-to-add-a-new-dataset)
  - [How to Add a New Model?](#how-to-add-a-new-model)
- [To do](#to-do)
- [Citations](#citations)

## Overview
[ColossalEval](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval) is a project which provides a uniform pipeline to help evaluate language models on different public dataset or your own dataset using both classic metrics and the help from GPTs. Currently we support AGIEval, CEval, CMMLU, CValues, GAOKAO-Bench, GSM8K, LongBench, MMLU, MtBench and SafetyBench. More details can be found in the following sections.

## Leaderboard
### Model with ~13 Billion Parameters
We conducted comprehensive evaluation on 5 datasets and compare our Colossal-Llama-2-13b-base model with various models.

- We use 5-shot for MMLU and calculate scores based on the logits of first predicted token.
- We use 5-shot for CMMLU and calculate scores based on the logits of first predicted token.
- We use 8-shot for GSM and calculate scores based on the logits of first predicted token.
- We use 5-shot for AGIEval and only calculate scores for 4-choice questions using a combination metric of exact match and the logits of first predicted token. If any of the exact match or logits of first predicted token is correct, the model will get the score.
- We use 0-shot for GAOKAO-Bench and only calculate scores for 4-choice questions based on the logits of first predicted token.
- The generation config for all dataset is greedy search.
- We also provided CEval scores from its latest leaderboard or the official repository of the model.

|                                 | Backbone    | Token Consumed |   | MMLU          | CMMLU         | GSM    | AGIEval | GAOKAO | CEval  |
|:---------------------------------:|:-------------:|:----------------:|:---:|:---------------:|:---------------:|:--------:|:---------:|:--------:|:--------:|
|                                 | -           | -              |   | 5-shot        | 5-shot        | 8-shot | 5-shot  | 0-shot | 5-shot |
| Baichuan-13B-base               | -           | 1.4T           |   | 50.54 (51.60) | 55.52 (55.30) |  25.78 |  41.86  |  51.62 |  53.60 |
| Baichuan2-13B-base              | -           | 2.6T           |   | 54.81 (59.17) | 62.68 (61.97) |  53.98 |  48.22  |  58.60 |  58.10 |
| InternLM-20B                    | -           | 2.3T           |   | 60.51 (62.05) |   59.46 (-)   |  51.4  |  56.07  |  62.06 |    -   |
| Qwen-14B                        | -           | 3.0T           |   |     66.51     |     71.08     |  61.33 |  66.62  |  80.82 |  72.1  |
| Skywork-13B-base                | -           | 3.2T           |   |     61.84     |     61.93     |  54.28 |  53.13  |  63.02 |    -   |
|                                 |             |                |   |               |               |        |         |        |        |
|           Llama-2-13B           |      -      |      2.0T      |   |     55.35     |     38.14     |  31.31 |  40.07  |  27.86 |    -   |
| Linly-AI/Chinese-LLaMA-2-13B-hf | Llama-2-13B |        -       |   |     51.82     |     42.73     |  36.01 |  39.47  |  28.28 |    -   |
|     hfl/chinese-llama-2-13b     | Llama-2-13B |        -       |   |     51.51     |     42.83     |  23.20 |  40.46  |  30.89 |    -   |
|  wenge-research/yayi-13b-llama2 | Llama-2-13B |        -       |   |      23.7     |     25.34     |  7.51  |  24.72  |  27.22 |    -   |
| TigerResearch/tigerbot-13b-base | Llama-2-13B |        0.6T       |   |     52.31     |     51.74     |  44.50 |  42.70  |  38.22 |    -   |
|     IDEA-CCNL/Ziya2-13B-Base    | Llama-2-13B |        0.65T       |   |     59.37     |     61.16     |  44.58 |  51.72  |  58.96 |    58.84   |
|                                 |             |                |   |               |               |        |         |        |        |
|    **Colossal-LLaMA-2-13b-base**    | Llama-2-13B |     **0.025T**     |   |     56.42     |      61.8     |  58.83 |  54.69  |  69.53 |  60.3  |

> The score in parentheses corresponds to the scores in the official repository of the model.

More details about metrics can be found in [Metrics](#metrics).

### Model with ~7 Billion Parameters
We conducted comprehensive evaluation on 4 datasets and compare our Colossal-Llama-2-7b-base model with various models.

- We use 5-shot for MMLU and calculate scores based on the logits of first predicted token.
- We use 5-shot for CMMLU and calculate scores based on the logits of first predicted token.
- We use 5-shot for AGIEval and only calculate scores for 4-choice questions using a combination metric of exact match and the logits of first predicted token. If any of the exact match or logits of first predicted token is correct, the model will get the score.
- We use 0-shot for GAOKAO-Bench and only calculate scores for 4-choice questions based on the logits of first predicted token.
- The generation config for all dataset is greedy search.
- We also provided CEval scores from its latest leaderboard or the official repository of the model.

More details about metrics can be found in [Metrics](#metrics).

|                                |  Backbone  | Tokens Consumed |  |         MMLU         |     CMMLU     | AGIEval | GAOKAO | CEval  |
| :----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :-----: | :----: | :----: | :----------------------------: |
|                                |     -      |        -        |                |        5-shot        |    5-shot     | 5-shot  | 0-shot | 5-shot |
|          Baichuan-7B           |     -      |      1.2T       |             |    42.32 (42.30)     | 44.53 (44.02) |  38.72  | 36.74  | 42.80  |
|       Baichuan2-7B-Base        |     -      |      2.6T       |             |    46.97 (54.16)     | 57.67 (57.07) |  45.76  | 52.60  | 54.00  |
|           ChatGLM-6B           |     -      |      1.0T       |             |    39.67 (40.63)     |   41.17 (-)   |  40.10  | 36.53  | 38.90  |
|          ChatGLM2-6B           |     -      |      1.4T       |             |    44.74 (45.46)     |   49.40 (-)   |  46.36  | 45.49  | 51.70  |
|          InternLM-7B           |     -      |        -        |                |    46.70 (51.00)     |   52.00 (-)   |  44.77  | 61.64  | 52.80  |
|            Qwen-7B (original)             |     -      |      2.2T       |             | 54.29 (56.70) | 56.03 (58.80) |  52.47  | 56.42  | 59.60  |
|            Qwen-7B             |     -      |      2.4T       |             | 58.33 (58.20) | 62.54 (62.20) |  64.34  | 74.05 | 63.50 |
|                                |            |                 |                 |                      |               |         |        |        |
|           Llama-2-7B           |     -      |      2.0T       |             |    44.47 (45.30)     |   32.97 (-)   |  32.60  | 25.46  |   -    |
| Linly-AI/Chinese-LLaMA-2-7B-hf | Llama-2-7B |      1.0T       |             |        37.43         |     29.92     |  32.00  | 27.57  |   -    |
| wenge-research/yayi-7b-llama2  | Llama-2-7B |        -        |                |        38.56         |     31.52     |  30.99  | 25.95  |   -    |
| ziqingyang/chinese-llama-2-7b  | Llama-2-7B |        -        |                |        33.86         |     34.69     |  34.52  | 25.18  |  34.2  |
| TigerResearch/tigerbot-7b-base | Llama-2-7B |      0.3T       |             |        43.73         |     42.04     |  37.64  | 30.61  |   -    |
|  LinkSoul/Chinese-Llama-2-7b   | Llama-2-7B |        -        |                |        48.41         |     38.31     |  38.45  | 27.72  |   -    |
|       FlagAlpha/Atom-7B        | Llama-2-7B |      0.1T       |             |        49.96         |     41.10     |  39.83  | 33.00  |   -    |
| IDEA-CCNL/Ziya-LLaMA-13B-v1.1  | Llama-13B  |      0.11T      |            |        50.25         |     40.99     |  40.04  | 30.54  |   -    |
|  |  |  |  |  |  |  |  |  |
|    **Colossal-LLaMA-2-7b-base**    | Llama-2-7B |      **0.0085T**      |            |        53.06         |     49.89     |  51.48  | 58.82  |  50.20  |

> The score in parentheses corresponds to the scores in the official repository of the model.
>
> We use zero-shot for ChatGLM models.
>
> To evaluate Qwen-7B on dataset MMLU, the prompt would be "xxx Answer:"(remove the space after ":") and we calculate the logits over " A", " B", " C" and " D" for Qwen-7B. Both the original and updated versions of Qwen-7B tend to be much more deterministic than other models. For example, the logits over " A" can be `-inf` and softmax would be exact `0`.
>
> For other models and other dataset, we calculate logits over "A", "B", "C" and "D".

Our model achieves a much better score over all other Llama-1 or Llama-2 based models and also stands out among popular open source LLMs.

## Install
You should install `ColossalEval` in order to use it and `colossal_eval` is the package installed.
```bash
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI/applications/ColossalEval
pip install .
```
If you want to add customized dataset or models, use `pip install -e .` in stead to ensure that any changes you make to the source code will immediately affect the package you install.

## Evaluation Process
The evaluation process involves 2 steps which are `inference` and `evaluation`. You need to set the config for each step.

### Inference

The inference process consists of two parts. We now support tensor parallel inference for large models using [ShardFormer](colossalai/shardformer) in the [example](applications/ColossalEval/examples/dataset_evaluation/inference.py) script.
1. Preprocess and convert the original dataset.
2. Config your tokenizer and model arguments to perform zero-shot or few-shot prompting.

#### Dataset Preparation

In this step, the original dataset(either in `csv` or `jsonl` format) will be loaded and converted into a `dict`. In the conversion process, we carefully parse each subcategory and assign specific inference arguments for this subcategory.

Inference arguments are stored in a `dict`. The following is an example.

```python
inference_kwargs = {
    "calculate_loss": True,
    "all_classes": ["A", "B", "C", "D"],
    "language": "Chinese",
    "calculate_overall_loss": False,
    "max_new_tokens": 32
}
```
The `inference_kwargs` currently contains 5 fields:

- `calculate_loss` (bool, compulsory): Whether the loss on target tokens will be calculated
- `all_classes` (Optional[list], compulsory): Whether the subcategory is a single-choice question. Specify all available options in a list or otherwise None.
- `language` (str, compulsory): The language for the subcategory.
- `calculate_overall_loss` (bool, compulsory): Whether to calculate the overall loss of sentences or not if the dataset is a pretrain dataset. It is usually used for calculate perplexity when you want to evaluate a model with extended context length.
- `max_new_tokens` (int, compulsory): The number of new tokens to generate during inference.

For example, for dataset MMLU, each subcategory consists of single-choice questions with options A, B, C and D by default and we can assign value `["A", "B", "C", "D"]` to key`all_classes`. For dataset C-Eval, target answers aren't provided in the test split so `calculate_loss` should be set as False. However, other dataset such as GAOKAO-bench contains different formats of questions and lacks some keys or metadata which can reveal what type (single-choice or multi-choice) of questions it is. Before assigning inference arguments, we first parse the dataset to decide which type of questions the subcategory belongs to and set the inference arguments accordingly.

Other than `inference_kwargs`, `data` is a list containing questions of a same subcategory. The following is a converted dataset.

```json
{
    "dev": {
        "category 1": {"data": [], "inference_kwargs": {}},
        "category 2": {"data": [], "inference_kwargs": {}}
    },
    "test": {
        "category 1": {"data": [], "inference_kwargs": {}},
        "category 2": {"data": [], "inference_kwargs": {}}
    }
}
```

A data sample basically follow the format of Alpaca. It should contain the following keys:

* `dataset` (str, compulsory): The name of the dataset.
* `split` (str, compulsory): The split of the instruction.
* `category` (str, compulsory): The category of the instruction.
* `instruction` (str, compulsory): The instruction for the LLM.
* `input` (str, optional): The additional context of the instruction.
* `output` (str, optional): The model output of the instruction.
* `target` (str, optional): The target answer for the instruction.

Example:

```json
{
    "dev": {
        "Abstract Algebra": [
            {
                "dataset": "mmlu",
                "split": "dev",
                "category": "Abstract Algebra",
                "instruction": "The following is a single-choice question on Abstract Algebra. Answer the question by replying A, B, C or D.",
                "input": "Question: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: ",
                "output": "",
                "target": "B"
            },
        ]
    },
    "test": {
        "Abstract Algebra": [
            {
                "dataset": "mmlu",
                "split": "test",
                "category": "Abstract Algebra",
                "instruction": "The following is a single-choice question on Abstract Algebra. Answer the question by replying A, B, C or D.",
                "input": "Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer: ",
                "output": "",
                "target": "B"
            },
        ]
    }
}
```

#### Configuration
In this step, you will configure your tokenizer and model arguments to infer on the given datasets.

A config file consists of two parts.
1. Model config. In model config, you need to specify model name, model path, model class, tokenizer arguments and model arguments. For model class, currently we support `HuggingFaceModel`, `HuggingFaceCausalLM`, `ChatGLMModel`, `ChatGLMModel2` and `vLLMModel`. `HuggingFaceModel` is for models that can be loaded with `AutoModel` and `HuggingFaceCausalLM` is for models that can be loaded with `AutoModelForCausalLM`. `ChatGLMModel` and `ChatGLMModel2` are for ChatGLM and ChatGLM2 models respectively. `vLLMModel` is for models that can be loaded with vllm offline inference `LLM` class. You can check all model classes in `colossal_eval/models/__init__.py`. If your model should set `trust_remote_code` as true, specify it in the `tokenizer_kwargs` and `model_kwargs` fields.
2. Dataset config. In dataset config, you need to specify dataset name, path and dataset class. Currently, we support zero-shot on dataset MMLU, CMMLU, AGIEval, GAOKAO-Bench, GSM8K and LongBench and few-shot on dataset MMLU, CMMLU AGIEval and GSM8K. If you want to enable few shot, set `few_shot` as true. You can check all model classes in `colossal_eval/dataset/__init__.py`.

Once you have all config ready, the program will run inference on all the given datasets on all the given models.

An example config using model class `HuggingFaceCausalLM` and dataset class `CMMLUDataset` can be:
```json
{
    "model": [
        {
            "name": "model name",
            "model_class": "HuggingFaceCausalLM",
            "parameters": {
                "path": "path to model",
                "model_max_length": 2048,
                "tokenizer_path": "path to tokenizer",
                "tokenizer_kwargs": {
                    "use_fast": false,
                    "trust_remote_code": true
                },
                "peft_path": null,
                "model_kwargs": {
                    "trust_remote_code": true
                },
                "prompt_template": "plain",
                "batch_size": 4
            }
        }
    ],
    "dataset": [
        {
            "name": "dataset name",
            "dataset_class": "CMMLUDataset",
            "debug": false,
            "few_shot": true,
            "path": "path to original dataset",
            "save_path": "path to save converted dataset"
        }
    ]
}
```

An example config using model class `vLLMModel` and dataset class `CMMLUDataset` can be:
```json
{
    "model": [
        {
            "name": "model name",
            "model_class": "vLLMModel",
            "parameters": {
                "path": "path to model",
                "model_max_length": 2048,
                "tokenizer_path": "",
                "tokenizer_kwargs": {
                    "trust_remote_code": true
                },
                "model_kwargs": {
                    "trust_remote_code": true
                },
                "prompt_template": "plain",
                "batch_size": 4
            }
        }
    ],
    "dataset": [
        {
            "name": "dataset name",
            "dataset_class": "CMMLUDataset",
            "debug": false,
            "few_shot": true,
            "path": "path to original dataset",
            "save_path": "path to save converted dataset"
        }
    ]
}
```

Currently, we support Hugging Face models as well as vLLM models. For Hugging Face models, the `tokenizer_kwargs` is the arguments used in `AutoTokenizer.from_pretrained()`. The `model_kwargs` is the arguments used in `AutoModel.from_pretrained` or `AutoModelForCausalLM.from_pretrained()`. For vLLM model, the `tokenizer_kwargs` and `model_kwargs` are loaded together in `LLM` class.`few_shot` will be set true if you want to enable few-shot prompting for the dataset. `debug` will be set true if you want to verify whether your prompt is right or wrong.

> For GSM8K dataset, you can set additional flags `load_train` or `load_reference` for dataset configuration as true and during the inference process, the program will calculate loss summation over all tokens for each data sample. During the evaluation process, you can use metric `loss_over_all_tokens` to calculate the overall loss and use it for data leakage evaluation.

#### How to Use
An example script can be the following. The `configs/dataset_evaluation/inference.py` is the same in all examples provided.

```shell
torchrun --nproc_per_node=4 inference.py \
    --config "path to config file" \
    --load_dataset \
    --tp_size 2 \
    --inference_save_path "path to save inference results"
```

You should specify the path to config file in `config`. You can run the script without specifying `load_dataset` if you already save the converted dataset or otherwise set it to first load the original dataset and save the converted dataset. You should specify the path to save inference results in `inference_save_path`. If you want to use tensor parallel inference, specify the tensor parallel size in `--tp_size` and the process will automatically calculate data parallel size (currently not support for `vLLMModel`).

### Evaluation

In the evaluation process, you only need to configure your evaluation parameters. You can use either public dataset or help from GPTs to do evaluation. We will introduce configuration for dataset evaluation and GPT evaluation.

#### Dataset Evaluation

In dataset evaluation, we calculate different metrics on the given inference results and public dataset.

##### Configuration

A config file for dataset evaluation consists of two parts.
1. Model config. In model config, you need to specify model name. If you want to evaluate perplexity over a pretrain dataset and calculate per-byte-perplexity, you have to add your tokenizer config and model max length.
2. Dataset config. In dataset config, you need to specify the evaluation metrics for the dataset.

Once you have all config ready, the program will run evaluation on inference results for all given models and dataset.

An example config can be:
```json
{
    "model": [
        {
            "name": "model name"
        }
    ],
    "dataset": [
        {
            "name": "dataset name",
            "metrics": ["first_token_accuracy"]
        }
    ]
}
```

The above config specifies that the program will evaluate the inference results using `first_token_accuracy` metric.

##### How to Use

An example script can be the following.

```shell
python eval_dataset.py \
    --config "path to config file" \
    --inference_results_path "path to inference results" \
    --evaluation_results_save_path "path to save evaluation results"
```

You should specify the path to config file in `config`, the path to inference results in `inference_results_path` and the path to save evaluation results in `evaluation_save_path`.

#### GPT Evaluation

In GPT evaluation, we provide a prompt template which can fit in different pre-defined metrics with Chain-of-Thoughts. In the following sections, we will only introduce how you can evaluate model answers using GPTs. More details can be found in `colossal_eval/evaluate/GPT Evaluation.md`.

##### Configuration

The following is an example of a English config file. The configuration file can control how the pipeline evaluates the model. You need to specify GPT evaluation metrics. You can find an example English config file in `configs/gpt_evaluation`.

```json
{
    "language": "en",
    "category": {
        "brainstorming": {
            "GPT": [
                "language organization",
                "relevance",
                "creativity",
                "practicality",
                "reasonableness"
            ]
        },
    }
}
```

##### How to Use
After setting the config file, you can evaluate the model using `examples/gpt_evaluation/eval.py`. If you want to make comparisons between answers of two different models, you should specify two answer files in the argument `answer_file_list` and two model names in the argument `model_name_list`(details can be found in `colossal_eval/evaluate/GPT Evaluation.md`). If you want to evaluate one answer file, the length of both `answer_file_list` and `model_name_list` should be 1 and the program will perform evaluation using GPTs. The prompt files for battle and gpt evaluation can be found in `configs/gpt_evaluation/prompt`. `target file` is the path to the converted dataset you save during inference time.

An example script is provided as follows:

```shell
python eval.py \
    --config_file "path to the config file" \
    --battle_prompt_file "path to the prompt file for battle" \
    --gpt_evaluation_prompt_file "path to the prompt file for gpt evaluation" \
    --target_file "path to the target answer file" \
    --answer_file_list "path to the answer file" \
    --model_name_list "the names of the model" \
    --gpt_model "which GPT model to use for evaluation" \
    --save_path "path to save results" \
    --openai_key "your openai key" \
```

## More Details

### Inference

In the inference process, we will do generation, calculate loss over target tokens, calculate number of target tokens, softmax over given options (for example, "A", "B", "C", and "D") according to the inference arguments.

For tokenization, we adopt tokenization strategy in [LongBench](https://github.com/THUDM/LongBench/blob/main/pred.py#L55) to preserve crucial instructions on the left and right side and keep all target tokens.

For labeling target tokens, we adopt method from [FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py#L137), but it doesn't always hold true due to tokenizers' different behavior. We plan to insert special tokens to correctly label the target tokens.

For calculating loss, we return per-sample-loss instead of per-batch-loss if we directly use `model(batch).loss` provided in HuggingFace.

### Evaluation

To make it more easier to set the config, you only need to specify all metrics you want to use in key `metrics`. However, the program will only use a subset of metrics you give for different subcategories. Applying all metrics to all subcategories is obviously unsuitable. The suggested metrics for specific categories should be defined in `colossal_eval/evaluate/dataset_evaluator/metrics.py`.

#### Metrics

- `combined_single_choice_accuracy`: A combination of `first_token_logit` and `single_choice_accuracy`. If one of these is correct, the model will get the score. It can be used in all dataset that contains single-choice questions.
- `first_token_logit`: Calculate score based on softmax score over the given choices. If the argmax of the softmax is equal to the reference, the model will get the score. If there is `NaN` in softmax score, it will calculate the score using exact match. It can be used in all dataset that contains single-choice questions.
- `single_choice_accuracy`: Calculate score using exact match. It will only get the first uppercase letter such as A, B, C or D that is not surrounded by lowercase letters. If the uppercase letter is equal to the reference, the model will get the score. It can be used in all dataset that contains single-choice questions.
- `multi_choice_accuracy`: Calculate score on multi-choice questions. It will get a set of all uppercase letters such as A, B, C or D that is not surrounded by lowercase letters. If the prediction contains uppercase letters that are not in reference. The model will get 0 score. If the prediction contains a uppercase letter that is in reference, the model will get a score of `1/len(reference)`. It is used in AGIEval and GAOKAO-Bench.
- `math_equivalence`: Code from [hendrycks](https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py). Compute scores over the prediction math formula and reference math formula. It is used in AGIEval and GAOKAO-Bench.
- `f1_score`: Calculate English f1 score between prediction and reference. It is used in Longbench.
- `f1_zh_score`: Calculate Chinese f1 score between prediction and reference. It is used in Longbench.
- `rouge_score`: Calculate English f1 score between prediction and reference. It is used in GAOKAO-Bench and LongBench.
- `rouge_zh_score`: Calculate Chinese rouge score between prediction and reference. It is used in GAOKAO-Bench and LongBench.
- `retrieval_score`: Calculate English retrieval score between prediction and reference. It determines whether the output(which paragraph) corresponds to the given abstract. It is used in Longbench.
- `retrieval_zh_score`: Calculate Chinese retrieval score between prediction and reference. It determines whether the output(which paragraph) corresponds to the given abstract. It is used in Longbench.
- `classification_score`: Calculate classification score between prediction and reference. It determines whether the output(a class) is equal to the reference. It is used in Longbench.
- `code_sim_score`: Calculate similarity score between prediction and reference. It is used in Longbench.
- `count_score`: Calculate count score between prediction and reference. It determines whether the output(number of given passages) is equal to the reference. It is used in Longbench.
- `gsm_accuracy`: Calculate scores between prediction and reference.. It is used in GSM8K.
- `perplexity`: Calculate perplexity. The formula is $ perplexity = \frac{1}{n} \sum_i e^{loss_i} $ where $n$ is the number of samples and $ loss_i $ is the average loss for sample $ i $. It can be used in all dataset.
- `ppl_score`: Calculate perplexity score. The formula is $ ppl\_score = \frac{1}{n} \sum_i e^{-loss_i} $ where $n$ is the number of samples and $ loss_i $ is the average loss for sample $ i $. It can be used in all dataset.
- `ppl_score_over_choices`: Calculate perplexity score over choices. The formula is $ ppl\_score\_over\_choices= \frac{1}{n} \sum_i e^{-loss\_over\_choices_i} $ where $n$ is the number of samples and $ loss\_over\_choices_i $ is the loss on the first predicted token for sample $ i $. It can be used in all dataset that contains single-choice questions.
- `per_byte_perplexity`: Calculate per byte perplexity. The formula is $ \frac{1}{n} \sum_i e^{\frac{loss_i}{byte_i}} $ where $n$ is the number of samples, $ loss_i $ is the total loss for sample $ i $ and $ byte_i $ is the number of bytes sample $ i $ occupies. It can be used in all dataset.
- `per_byte_ppl_score`: Calculate per byte perplexity score. The formula is $ \frac{1}{n} \sum_i e^{-\frac{loss_i}{byte_i}} $ where $n$ is the number of samples, $ loss_i $ is the total loss for sample $ i $ and $ byte_i $ is the number of bytes sample $ i $ occupies. It can be used in all dataset.
- `loss_over_all_tokens`: Calculate loss over all tokens. The formula is $ loss\_over\_all\_tokens = \frac{1}{n} \sum_i loss_i $ where $n$ is the total number of tokens of the dataset and $ loss_i $ is the loss summation for sample $ i $ over all tokens and $ \sum_i loss_i $ is the loss summation for all samples. It can be used in all dataset.

We use `combined_single_choice_accuracy` and `first_token_logit` in the leaderboard.

### Examples

We provide 2 examples for you to explore our `colossal_eval` package.

#### Dataset Evaluation Example

This example is in folder `examples/dataset_evaluation`.

1. `cd examples/dataset_evaluation`
2. Fill in your inference config file in `config/inference/config.json`. Set the model and dataset parameters.
3. Run `inference.sh` to get inference results.
4. Fill in your evaluation config file in `config/evaluation/config.json`. Set the model and dataset parameters.
5. Run `eval_dataset.sh` to get evaluation results.

#### GPT Evaluation Example

The examples is in folder `examples/gpt_evaluation`.

1. `cd examples/gpt_evaluation`
2. Fill in your inference config file in `config/inference/config.json`. Set the model and dataset parameters. If you want to use the example dataset we provide, the dataset is `ColossalDataset`.
3. Run `inference.sh` to get inference results.
4. Fill in your evaluation config file in `config/evaluation/config.json`.
5. Run `eval.sh` to get evaluation results.

## FAQ

### How to Add a New Metric?

If you want to add a customized metric, we recommend using `pip install -e .` to ensure that any changes you make to the source code will immediately affect the package you install.

To add a new metric, you can follow the example of multi_choice_accuracy in line 339 in `colossal_eval/evaluate/dataset_evaluator/metric.py`. The method take one data sample's prediction and reference as input and return a score ranging from 0 to 1.

A skeleton of code is the following.

```python

def CustomizedMetric(prediction: str, reference: str):
	score = xxx
	return score
```

Once you have successfully added your own metric, you should specify your metric both in `colossal_eval/evaluate/dataset_evaluator/metric.py` (suggest which subcategories should the metric be applied to) and your evaluation config.

### How to Add a New Dataset?

If you want to add customized dataset, we recommend using `pip install -e .` to ensure that any changes you make to the source code will immediately affect the package you install.

To add a new dataset, you can follow the example of `colossal_eval/dataset/mmlu.py`. You need to make sure that the format of questions in one subcategory should be the same. For example, all questions should have target answers or all questions should be single-choice questions.

A skeleton of code is the following.

```python

class CustomizedDataset(BaseDataset):
    @staticmethod
    def load():
        # 1. Load and convert the original dataset format.
    	# 2. Assign inference arguments for each subcategory.
    	# 3. Return the converted dataset.
    	pass
```

Once you have successfully added your own dataset, you can specify your dataset class in your inference config.

### How to Add a New Model?

If you want to add customized models, we recommend using `pip install -e .` to ensure that any changes you make to the source code will immediately affect the package you install.

To add a new model, you can follow the example of `colossal_eval/models/huggingface.py`. You need to provide a way to load the model and tokenizer, calculate loss and generate.

A skeleton of code is the following.

```python

class CustomizedModel(BaseModel):
    def __init__(self):
        super().__init__()
		self._load_tokenizer()
		self._load_model()

	def _load_tokenizer():
		pass

	def _load_model():
		pass

	def _calculate_loss():
		pass

	def get_loss():
		self._calculate_loss()

	def inference(samples):
		# 1. Load samples from the same subcategory.
		# 2. Infer in a batch way according to inference arguments.
		# 3. Return results.
		batch_samples = xxx
		self.get_loss(batch_samples)
		self.generate(batch_samples)

		return inference_results

	def generate():
		pass
```

Once you have successfully added your own model, you can specify your model class in your inference config.


## Citations

```bibtex
@misc{zhong2023agieval,
      title={AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models},
      author={Wanjun Zhong and Ruixiang Cui and Yiduo Guo and Yaobo Liang and Shuai Lu and Yanlin Wang and Amin Saied and Weizhu Chen and Nan Duan},
      year={2023},
      eprint={2304.06364},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{huang2023ceval,
title={C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models},
author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian},
journal={arXiv preprint arXiv:2305.08322},
year={2023}
}

@misc{li2023cmmlu,
      title={CMMLU: Measuring massive multitask language understanding in Chinese},
      author={Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin},
      year={2023},
      eprint={2306.09212},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{xu2023cvalues,
      title={CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility},
      author={Guohai Xu and Jiayi Liu and Ming Yan and Haotian Xu and Jinghui Si and Zhuoran Zhou and Peng Yi and Xing Gao and Jitao Sang and Rong Zhang and Ji Zhang and Chao Peng and Fei Huang and Jingren Zhou},
      year={2023},
      eprint={2307.09705},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{Zhang2023EvaluatingTP,
  title={Evaluating the Performance of Large Language Models on GAOKAO Benchmark},
  author={Xiaotian Zhang and Chunyang Li and Yi Zong and Zhengyu Ying and Liang He and Xipeng Qiu},
  year={2023}
}

@misc{bai2023longbench,
      title={LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding},
      author={Yushi Bai and Xin Lv and Jiajie Zhang and Hongchang Lyu and Jiankai Tang and Zhidian Huang and Zhengxiao Du and Xiao Liu and Aohan Zeng and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
      year={2023},
      eprint={2308.14508},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{zhang2023safetybench,
      title={SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions},
      author={Zhexin Zhang and Leqi Lei and Lindong Wu and Rui Sun and Yongkang Huang and Chong Long and Xiao Liu and Xuanyu Lei and Jie Tang and Minlie Huang},
      journal={arXiv preprint arXiv:2309.07045},
      year={2023}
}

@article{cobbe2021training,
  title={Training verifiers to solve math word problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{wei2023skywork,
      title={Skywork: A More Open Bilingual Foundation Model},
      author={Tianwen Wei and Liang Zhao and Lichang Zhang and Bo Zhu and Lijie Wang and Haihua Yang and Biye Li and Cheng Cheng and Weiwei Lü and Rui Hu and Chenxia Li and Liu Yang and Xilin Luo and Xuejie Wu and Lunan Liu and Wenjun Cheng and Peng Cheng and Jianhao Zhang and Xiaoyu Zhang and Lei Lin and Xiaokun Wang and Yutuan Ma and Chuanhai Dong and Yanqi Sun and Yifu Chen and Yongyi Peng and Xiaojuan Liang and Shuicheng Yan and Han Fang and Yahui Zhou},
      year={2023},
      eprint={2310.19341},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
