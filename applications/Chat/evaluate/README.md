# Evaluation

In this directory we will introduce how you can evaluate your model with GPT-4. 

## Evaluation Pipeline

The whole evaluation process undergoes two steps. 

1. Generate answers from different models: Use `generate_gpt35_answers.py` to generate answers of GPT 3.5 and use `generate_answers.py` to generate answers of your own models.
2. Evaluate models using GPT 4: Use `evaluate.py` to evaluate model answers with GPT-4.

### Generate Answers

To generate answers, you should first format [FastChat's]([FastChat/question.jsonl at main · lm-sys/FastChat (github.com)](https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/question.jsonl)) `question.jsonl` file. We do this formatting because we would like to add more questions later and the pipeline for generating new questions may follow that of Self-Instruct and Stanford Alpaca. An example script is given as follows.

```shell
python format_questions.py \
    --questions_path "path to FastChat's question.jsonl" \
    --save_path "path to the formatted file" \

```

In `generate_answers.py`, the model will generate answers in a batch way and different GPU processes will do inference on different shards of the given questions. Once all GPU process generate its answers, `merge.py` will merge different shards of answers and output a single answer file. Finally, the script will also remove the answer shards. An example script is given as follows.

```shell
device_number=number of your devices
model_name="name of your model"
model_path="path to your model"
dataset="path to the question dataset"
answer_path="path to save the model answers"

torchrun --standalone --nproc_per_node=$device_number generate_answers.py \
    --model 'llama' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 8 \
    --max_datasets_size 80 \
    --answer_path $answer_path \
    --max_length 512

python merge.py \
    --model_name $model_name \
    --shards $device_number \
    --answer_path $answer_path \

for (( i=0; i<device_number; i++ )) do
    rm -rf "${answer_path}/${model_name}_answers_rank${i}.json"
done

```

`generate_gpt35_answers.py` will generate answers of GPT-3.5 An example script is given as follows.

```shell
python generate_gpt35_answers.py \
    --dataset "path to the question dataset" \
    --answer_path "path to answer folder" \
    --num_workers 4 \
    --openai_key "your openai key" \
    --max_tokens 512 \

```

### Evaluate Answers

In `evaluate.py`, GPT-4 will help review and score answers of two different models. Here `Model 1` refers to the first model you specify in the `--answer_file_list` and `Model 2` refers to the second model. The script will finally print several metrics and output corresponding JSON files.

The metrics include:

- `Invalid Count`: The number of reviews where the program fail to parse the score pair.
- `Better Count`: The number of reviews where Model 2 receives a higher score.
- `Worse Count`: The number of reviews where Model 2 receives a lower score.
- `Tie Count`: The number of reviews where two models play to a tie.
- `Win Rate of Model 2`: Win rate of Model 2.
- `Model 1 Average Score`: Average score of Model 1.
- `Model 2 Average Score`: Average score of Model 2.

Other than the `review` and `result` file which include all reviews, the output files also include `invalid`, `better`, `worse` and `tie` JSON file which only include the corresponding reviews.

```shell
python evaluate.py \
    --answer_file_list "path to answers of model 1" "path to answers of model 2" \
    --prompt_file "path to prompt file" \
    --reviewer_file "path to reviewer file" \
    --output_folder "path to output folder" \
    --openai_key "your openai key" \
    --model "the gpt model" \
    --num_workers 8 \
    --max_tokens 512 \

```

## Results

We compare our model with alpaca and vicuna. The results is shown below. Please note that the better cases don't add to 80 because there are reviews the program can't successfully parse to get the score pair. Our Coati-7B model performs better than Alpaca-7B. The Coati-7B model we evaluate is an old version we trained a few weeks ago and the new version is around the corner.

|  Model Pair   | Alpaca-7B ⚔ Coati-7B | Coati-7B ⚔ Alpaca-7B |
| :-----------: | :------------------: | :------------------: |
| Better Cases  |     38 ⚔ **41**      |     **45** ⚔ 33      |
|   Win Rate    |    48% ⚔ **52%**     |    **58%** ⚔ 42%     |
| Average Score |   7.06 ⚔ **7.13**    |   **7.31** ⚔ 6.82    |

We would like to mention that the evaluation of model answers using the GPT-3.5 model is not reliable. GPT-3.5 tends to give a higher score to the second answer (`{answer2}` in the prompt). In our evaluation which uses GPT-4, we still swap the two model answers. As can be seen from the table, GPT-4 can generate consistent results and it is more unbiased than GPT-3.5.

## Data Format

### Questions

We store questions in `questions.json`. The JSON file contains one list. Each element in the list is a question record.

A question record has the following field:

* `category` (str): The category of the question.
* `instruction` (str): The question.
* `input` (str): This is empty if you only use [FastChat's]([FastChat/question.jsonl at main · lm-sys/FastChat (github.com)](https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/question.jsonl)) questions.
* `output` (str): This is empty.
* `id` (int): The question id.

### Answers

We store model answers in `{model_name}_answers.json`. The JSON file contains one list. Each element in the list is an answer record to one question.

An answer record has the following field:

* `category` (str): The category of the question.
* `instruction` (str): The question.
* `input` (str): This is empty if you only use [FastChat's]([FastChat/question.jsonl at main · lm-sys/FastChat (github.com)](https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/question.jsonl)) questions.
* `output` (str): The answer to the question.
* `id` (int): The question id.

### Results

We store evaluation results in `results.json`. The JSON file contains one dictionary. The key in the dictionary is formatted as `{model 1}_vs_{model 2}` and the value is also a dictionary contains metrics about the evaluation.

The value has the following field:

* `model` (list): The names of the two models.
* `better` (int): The number of reviews where Model 2 receives a higher score.
* `worse` (int): The number of reviews where Model 2 receives a lower score.
* `tie` (int): The number of reviews where two models play to a tie.
* `win_rate` (float): Win rate of Model 2.
* `score` (list): Average score of the two models.

### Better, Worse, Tie, Invalid, Review

To help better compare the model answers, we store JSON files whose name ends with `_better`, `_worse`, `_tie`, `_invalid` or `_review`. Each JSON file contains one list. Each element in the list is a record of better, worse, tie, invalid or all cases.

A record has the following field:

* `review_id` (str): Random UUID, not in use.
* `id` (int): The question id.
* `reviewer_id` (int): A unique ID for a reviewer. Different reviewer id use different prompts.
* `metadata` (dict): It is empty.
* `review` (str): GPT-4 's review.
* `score` (list): The scores of two models.

### Prompts

The data format is the same with [FastChat's]([FastChat/prompt.jsonl at main · lm-sys/FastChat (github.com)](https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/prompt.jsonl)) prompts.

### Reviewer

The data format is the same with [FastChat's]([FastChat/reviewer.jsonl at main · lm-sys/FastChat (github.com)](https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/reviewer.jsonl)) reviewers.

## Plan

- [ ] Extend the questions

## Citations

```bibtex
@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://vicuna.lmsys.org},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}
}
```
