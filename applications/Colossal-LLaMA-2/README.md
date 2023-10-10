<div align="center">
<h1>
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/colossalllam2.jpg?raw=true" width=800/>
</h1>
</div>

## Table of Contents
- [News](#news)
- [Colossal-LLaMA-2-7B](#colossal-llama-2-7b)
    - [Performance Evaluation](#performance-evaluation)
    - [Examples](#examples)
    - [Training Logs](#training-logs)
    - [Import from Transformers](#import-from-transformers)
- [Usage](#usage)
    - [Install](#install)
    - [How to run](#how-to-run)
- [Technical Insight](#technical-insights)
    - [Data](#data)
    - [Tokenizer](#tokenizer)
    - [Training Strategy](#training-strategy)
    - [Bridging Any Domain-specific Large Models](#bridging-any-domain-specific-large-models)
- [Citations](#citations)

## News
* [2023/09] [One Half-Day of Training Using a Few Hundred Dollars Yields Similar Results to Mainstream Large Models, Open-Source and Commercial-Free Domain-Specific Llm Solution](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution)
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA-2)
[[blog]](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution)
[[HuggingFace model weights]](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-7b-base)
[[Modelscope model weights]](https://www.modelscope.cn/models/colossalai/Colossal-LLaMA-2-7b-base/summary)


## Colossal-LLaMA-2-7B
The [Colossal-AI](https://github.com/hpcaitech/ColossalAI) team has introduced the open-source model **Colossal-LLaMA-2-7B-base**. This model, a derivation of LLaMA-2, has undergone continual pre-training involving approximately 8.5 billion tokens over a duration of 15 hours with 64 A800 GPUs. At a cost of **less than $1,000**, you can achieve results **similar to those that cost millions of dollars to pretrain from scratch**. It is licensed under the LLaMA-2 license and [Apache 2.0 License](https://github.com/hpcaitech/ColossalAI/blob/main/LICENSE) **without any additional commercial use restrictions**. This solution can also be used to build models of specific domain knowledge or tasks.

Colossal-LLaMA-2-7B-base is designed to accommodate both the Chinese and English languages, featuring an expansive context window spanning 4096 tokens. Remarkably, it has exhibited exceptional performance when benchmarked against models of equivalent scale in standard Chinese and English evaluation metrics, including C-Eval and MMLU, among others.

❗️**Important notice**:
* All training data used for this project is collected from well-known public dataset.
* We do not use any testing data from the evaluation benchmarks for training.

### Performance Evaluation
We conducted comprehensive evaluation on 4 dataset and compare our Colossal-Llama-2-7b-base model with various models.

* We use 5-shot for MMLU and calculate scores based on the logits of first predicted token.
* We use 5-shot for CMMLU and calculate scores based on the logits of first predicted token.
* We use 5-shot for AGIEval and only calculate scores for 4-choice questions using a combination metric of exact match and the logits of first predicted token. If any of the exact match or logits of first predicted token is correct, the model will get the score.
* We use 0-shot for GAOKAO-Bench and only calculate scores for 4-choice questions based on the logits of first predicted token.
The generation config for all dataset is greedy search.
* We also provided CEval scores from its lastest leaderboard or the official repository of the model.

|                                |  Backbone  | Tokens Consumed |  |         MMLU         |     CMMLU     | AGIEval | GAOKAO | CEval  |
| :----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :-----: | :----: | :----: | :------------------------------: |
|                                |           |        -        |                |        5-shot        |    5-shot     | 5-shot  | 0-shot | 5-shot |
|          Baichuan-7B           |     -      |      1.2T       |             |    42.32 (42.30)     | 44.53 (44.02) |  38.72  | 36.74  | 42.80  |
|       Baichuan-13B-Base        |     -      |      1.4T       |             |    50.51 (51.60)     | 55.73 (55.30) |  47.20  | 51.41  | 53.60  |
|       Baichuan2-7B-Base        |     -      |      2.6T       |             |    46.97 (54.16)     | 57.67 (57.07) |  45.76  | 52.60  | 54.00  |
|       Baichuan2-13B-Base       |     -      |      2.6T       |             |    54.84 (59.17)     | 62.62 (61.97) |  52.08  | 58.25  | 58.10  |
|           ChatGLM-6B           |     -      |      1.0T       |             |    39.67 (40.63)     |   41.17 (-)   |  40.10  | 36.53  | 38.90  |
|          ChatGLM2-6B           |     -      |      1.4T       |             |    44.74 (45.46)     |   49.40 (-)   |  46.36  | 45.49  | 51.70  |
|          InternLM-7B           |     -      |      1.6T       |                |    46.70 (51.00)     |   52.00 (-)   |  44.77  | 61.64  | 52.80  |
|            Qwen-7B (original)             |     -      |      2.2T       |             | 54.29 (56.70) | 56.03 (58.80) |  52.47  | 56.42  | 59.60  |
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
|    **Colossal-LLaMA-2-7b-base**    | Llama-2-7B |      **0.0085T**      |            |        53.06         |     49.89     |  51.48  | 58.82  |  50.2  |

> The score in parentheses corresponds to the scores in the official repository of the model.
>
> We use zero-shot for ChatGLM models.
>
> Qwen-7B is now inaccessible in Hugging Face, we are using the latest version of it before it was made inaccessible. Only for dataset MMLU, the prompt would be "xxx Answer:"(remove the space after ":") and we calculate the logits over " A", " B", " C" and " D" for Qwen-7B. Qwen-7B tends to be much more deterministic than other models. For example, the logits over " A" can be `-inf` and softmax would be exact `0`.
>
> For other models and other dataset, we calculate logits over "A", "B", "C" and "D".

❗️ More details of the evaluation methods and reproduction of the results, please refer to [ColossalEval](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval).

### Examples
| Question Type | Question | <center>Colossal-LLaMA-2-7b-base</center> |
| :------: | :-----------------------: | :----------- |
| Knowledge Quiz<br>(Tourism)  | 请问武汉的最佳的旅行季节是什么时候？ | 3-4月份到武汉赏樱，10月份到武汉看黄陂枫树。武汉的最佳的旅行季节是春秋，3-4月份到武汉赏樱，10月份到武汉看黄陂枫树。武汉，简称“汉”别称江城，是湖北省省会，中部六省唯一的副省级市，特大城市，中国中部地区的中心城市，全国重要的工业基地、科教基地和综合交通枢纽。 |
| Knowledge Quiz<br>(Computer Science)  | 在计算机中，什么是RAM？与ROM有什么区别？  | RAM是随机存取内存，可以随时读写，但是断电之后会丢失数据。存储数据比较快，但是需要不断的进行通电供电。ROM是只读内存，只能读取数据，不能写入数据。断电之后，存储的数据也不会丢失，但是需要不断进行通电。两者的价格和性能上会有区别，而且在使用的时候，RAM需要不断的供电，而ROM断电数据不丢失。  |
| Knowledge Quiz<br>(Literature)  | 古代文学与现代文学有哪些区别和共通之处？  | 古代文学与现代文学有三点区别，现代文学更加通俗易懂，具有可读性，而古代文学相对来说更加注重形式，缺乏可读性。现代文学和古代文学，同属于中华文学大花园中的一枝花蕾。但是它们由于历史的原因，有着不同的特点，具体表现在，现代文学有更丰富的描写方法，更强烈的主观色彩和更突出的抒情色彩，而古代文学的主要特征是在创作风格上的多样化。  |
| Emotional Analysis  | 请判断这个句子的情绪是否正面：我迫不及待想见到她！！！”。  | 这一句是情绪是正向的。是在期待见面的喜悦情绪。  |
| Translation  | 将以下句子翻译成英语。 我喜欢看电影和读书。  | I like watching movies and reading books.   |
| Logical Reasoning  | All books have the same weight, 10 books weigh 5kg, what is the weight of 2 books?  | To find the weight of 2 books, we can use the following steps:<br><br>1. Determine the weight of 1 book.<br>2. Multiply the weight of 1 book by the number of books.<br><br>Step 1: Determine the weight of 1 book.<br>Since 10 books weigh 5kg, we can divide 5kg by 10 to find the weight of 1 book.<br>5kg ÷ 10 = 0.5kg<br><br>Step 2: Multiply the weight of 1 book by the number of books.<br>To find the weight of 2 books, we multiply the weight of 1 book by 2.<br>0.5kg × 2 = 1kg<br><br>So, the weight of 2 books is 1kg. |
| Information Extraction  | The company's annual conference, featuring keynote speakers and exclusive product launches, will be held at the Los Angeles Convention Center from October 20th to October 23rd, 2021. Extract the date mentioned in the above sentence.  | The date mentioned in the sentence is October 20th to October 23rd, 2021.  |
| Error Correction  | Please correct the syntax error and restructure the sentence: "Went to the store me and my friend."  | The given sentence "Went to the store me and my friend." is incorrect due to a syntax error and it lacks a subject. The corrected version with proper grammar and sentence structure would be: "My friend and I went to the store."  |

❗️ More examples of question answering, please refer to [Colossal-LLaMA-2-7B-base Examples](docs/example.md).

### Training Logs
We also recorded the training logs for the experiment

<p id="Colossal-LLaMA-2-Multi-stage-training" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/trainingLossBySteps.jpeg?raw=true" width=600/>
</p>

<p id="Colossal-LLaMA-2-Multi-stage-training" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/trainingLossByTokens.jpeg?raw=true" width=600/>
</p>

### Import from Transformers (Inference)
To load Colossal-LLaMA-2-7B-base model using Transformers, use the following code:
```Python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("hpcai-tech/Colossal-LLaMA-2-7b-base", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("hpcai-tech/Colossal-LLaMA-2-7b-base", trust_remote_code=True)
input = "离离原上草，"
inputs = tokenizer(input, return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[len(input):])
```

You can also load our model using modelscope, use the following code:
```Python
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
model_dir = snapshot_download('colossalai/Colossal-LLaMA-2-7b-base', revision='v1.0.1')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
generation_kwargs = {"max_new_tokens": 256, 
                     "top_p": 0.95, 
                     "temperature": 0.3
                    }
input = '离离原上草，'
inputs = tokenizer(input, return_token_type_ids=False, return_tensors='pt')
inputs = inputs.to('cuda:0')
output = model.generate(**inputs, **generation_kwargs)
print(tokenizer.decode(output.cpu()[0], skip_special_tokens=True)[len(input):])
```
You can download model weights from [🤗HuggingFace](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-7b-base) or [👾Modelscope](https://modelscope.cn/models/colossalai/Colossal-LLaMA-2-7b-base/summary).

## Usage
### Install

#### 0. Pre-requisite
1. This experiment was performed on 8 computing nodes with 64 A800 GPUs in total for LLaMA-2-7B (**about 1000 USD cost**). The nodes are connected with RDMA and GPUs within one node are fully connected with NVLink. The script was tested with CUDA 11.7, CUDA version requires 11.7 or higher. You can also complete it in about 5 days on a 8*A100/A800 server.

2. PyTorch. The PyTorch version should be less than 2.0.0 and greater than 1.12.1.


#### 1. Install required packages
```
cd Colossal-LLaMA-2
pip install -r requirements.txt
```
#### 2. Install `xentropy`, `layer_norm` and `rotary`
```bash
git clone git@github.com:Dao-AILab/flash-attention.git
# At the root folder
cd csrc/xentropy && pip install .
# At the root folder
cd csrc/layer_norm && pip install .
# At the root folder
cd csrc/rotary && pip install .
```

### How to run

#### 1. Init Tokenizer Preparation
Initialize new tokenizer with additional Chinese tokens. Additional Chinese tokens are stored in `jsonl` format as follows:
```json
{"piece": "你好"}
{"piece": "人工智能"}
```
Command to initialize new tokenizer:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
python colossal_llama2/tokenizer/init_tokenizer.py \
    --source_tokenizer_dir "<SOURCE_TOKENIZER_DIR>" \
    --target_tokenizer_dir "<TARGET_TOKENIZER_DIR>" \
    --expand_tokens_file "<NEW_TOKENS_FILE>.jsonl"
```
Here is details about CLI arguments:
* Source tokenizer directory: `--source_tokenizer_dir`. Directory to the source tokenizer. It should at least contain three files: `special_tokens_map.json`, `tokenizer.model` and `tokenizer_config.json`.
* Target tokenizer directory: `--target_tokenizer_dir`. Directory to the target tokenizer.
* Tokens to be added: `--expand_tokens_file`. Additional tokens to be added to the tokenizer.

#### 2. Init Model Preparation
Initialize the new model checkpoint by calculating the mean values from the original model checkpoint.
Command to initialize new model checkpoint:
```bash
python colossal_llama2/model/init_model.py \
    --source_model_and_tokenizer_path "<SOURCE_MODEL_AND_TOKENIZER_DIR>" \
    --target_tokenizer_path "<TARGET_TOKENIZER_DIR>" \
    --target_model_path "<TARGET_MODEL_DIR>"
```
"<TARGET_MODEL_DIR>" can be the same as "<TARGET_TOKENIZER_DIR>".

Here is details about CLI arguments:
* Source model and tokenizer path: `--source_model_and_tokenizer_path`. Source folder contains both model and tokenizer, for example, LLaMA-2 model in Hugging Face format.
* Target tokenizer path: `--target_tokenizer_path`. Path to the new tokenizer folder generated from previous step.
* Target model path: `--target_model_path`. Path to save the new model in Hugging Face format.

❗️**Important**: Once you initialize the new model checkpoint, copy your new tokenizer files (`special_tokens_map.json`, `tokenizer.model` and `tokenizer_config.json`) to your new model folder.

#### 3. Data Preparation
Raw data should be formatted as `jsonl` format. Each data point should have the following fields:
* `source` (str, compulsory): This part is ignored when calculating loss. Default can be empty.
* `target` (str, compulsory): Loss will be calculated.
* `category` (str, compulsory): Tags for each data point.

Examples:
```JSON
{"source": "", "target": "Lionel Andrés Messi(Spanish pronunciation: [ljoˈnel anˈdɾes ˈmesi] (i); born 24 June 1987), also known as Leo Messi, is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team.", "category": "sports"}
{"source": "猜谜语：一身卷卷细毛，吃的青青野草，过了数九寒冬，无私献出白毛。（打一动物）", "target": "白羊", "category": "riddle"}
```
You are allowed to customize the category tags or use `unknown` to define the category.

Command to convert jsonl dataset to arrow format:
```
python prepare_pretrain_dataset.py \
    --data_input_dirs "<JOSNL_DIR_1>,<JOSNL_DIR_2>,<JOSNL_DIR_3>" \
    --tokenizer_dir "<TOKENIZER_DIR>" \
    --data_cache_dir "jsonl_to_arrow_cache" \
    --data_jsonl_output_dir "spliced_tokenized_output_jsonl" \
    --data_arrow_output_dir "spliced_tokenized_output_arrow" \
    --max_length 4096 \
    --num_spliced_dataset_bins 10
```
Here is details about CLI arguments:
* Source data directory: `data_input_dirs`. Each `<JOSNL_DIR>` can have multiple file in `jsonl` format.
* Tokenzier directory: `tokenizer_dir`. Path to the tokenizer in Hugging Face format.
* Data cache directory: `data_cache_dir`. Directory to store Hugging Face data cache. Default case will create `cache` folder locally.
* Output directory for jsonl format: `data_jsonl_output_dir`. Output directory to store converted dataset in jsonl format.
* Output directory for arrow format: `data_arrow_output_dir`. Output directory to store converted dataset in arrow format, which can be used for training directly.
* Max length: `max_length`. Max length of spliced samples. Default value is 4096.
* Number of bins for each category: `num_spliced_dataset_bins`. Number of bins for each category, used for bucket-based training.

#### 4. Command Line Arguments for Training
You can use `colossalai run` to launch multi-nodes training:
```bash
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE \
train.py --OTHER_CONFIGURATIONS
```
Here is a sample hostfile:
```bash
hostname1
hostname2
hostname3
hostname4
```
Make sure master node can access all nodes (including itself) by ssh without password.

Here is details about CLI arguments:
* Pre-trained model path: `--pretrained`. Path to the pre-trained model in Hugging Face format.
* Dataset path: `--dataset`. Path to the pre-tokenized dataset.
* Booster plugin: `--plugin`. `gemini`, `gemini_auto`, `zero2`，`zero2_cpu` and `3d` are supported.For more details, please refer to [Booster plugins](https://colossalai.org/docs/basics/booster_plugins/).
* Intermediate checkpoint to load: `--load_checkpoint`. Path to the intermediate checkpoint. Saved checkpoint contains the states for `lr_scheduler`, `optimizer`,`running_states.json` and `modelling`. If `load_checkpoint` points to the `modelling` folder, only the model weights will be loaded without any other states to support multi-stage training.
* Save interval: `--save_interval`. The interval (steps) of saving checkpoints. The default value is 1000.
* Checkpoint directory: `--save_dir`. The directoty path to save checkpoint and intermediate states. Intermediate states include `lr_scheduler`, `optimizer`,`running_states.json` and `modelling`.
* Tensorboard directory: `--tensorboard_dir`. The path to save tensorboard logs.
* Configuration file: `--config_file`. The path to save the configuration file.
* Number of epochs: `--num_epochs`. Number of training epochs. The default value is 1.
* Micro batch size: `--micro_batch_size`. Batch size per GPU. The default value is 1.
* Learning rate: `--lr`. The default value is 3e-4.
* Max length: `--max_length`. Max context length. The default value is 4096.
* Mixed precision: `--mixed_precision`. The default value is "fp16". "fp16" and "bf16" are supported.
* Gradient clipping: `--gradient_clipping`. The default value is 1.0.
* Weight decay: `-w`, `--weight_decay`. The default value is 0.1.
* Warmup steps: `-s`, `--warmup_steps`. The default value is calcuated by 0.025 warmup ratio.
* Gradient checkpointing: `--use_grad_checkpoint`. The default value is `False`. This saves memory at the cost of speed. You'd better enable this option when training with a large batch size.
* Flash attention: `--use_flash_attn`. If you want to use flash attention, you must install `flash-attn` and related packages. The default value is `False`. This is helpful to accelerate training while saving memory. We recommend you always use flash attention.
* Freeze non-embedding parameters: `--freeze_non_embeds_params`. Freeze non-embedding parameters. It can be helpful to align embeddings after extending vocabulary size.
* Tensor parallelism size: `--tp`. TP size for 3d Parallelism. The default value is 1.
* Zero stage: `--zero`. Zero stage for 3d Parallelism. The default value is 1.

#### 5. Running Command
An [example bash](train.example.sh) is also provided for the experiment. Here is the steps to run the experiment:
* Create your own hostfile: `cp hostfile.example hostfile`.
* Create your own bash: `cp train.example.sh train.sh`.
* Add your real host ip or host name into the `hostfile`.
* Update global variables and parameters in your `train.sh`.
* Run the experiment by `bash train.sh`

Here is the details about global variables for each experiment:
* `PROJECT_NAME`: Project name for each experiment.
* `PARENT_SAVE_DIR`: Parent folder to save model checkpoint.
* `PARENT_TENSORBOARD_DIR`: Parent folder to save tensorboard logs.
* `PARENT_CONFIG_FILE`: Parent folder to save configuration for each experiment.
* `PRETRAINED_MODEL_PATH`: Path to the local pre-trained model checkpoint.
* `dataset`: Paths to all prepared data. Typically, it's a list of subfolders within the output path of prepare data, `--data_arrow_output_dir`, and if there are multiple subfolders, please list them all. e.g.,
```python
declare -a dataset=(
    "<DIR_1>/part-00000"
    "<DIR_1>/part-00001"
    "<DIR_2>/part-00000"
)
```
## Technical Insights
In order to enhance LLaMA-2's capabilities for understanding and generating Chinese content, The [Colossal-AI](https://github.com/hpcaitech/ColossalAI) team proposes the continuation of pre-training the LLaMA-2 model using both Chinese and English corpora. The overall pipeline can be described as follows:

<p id="Colossal-LLaMA-2-pipeline" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/Colossal-LLaMA-2-pipeline.jpeg?raw=true" width=800/>
</p>

### Data
Large language models such as LLaMA-2 have undergone training using a heterogeneous blend of high-quality datasets, yielding promising outcomes. Enhancing LLaMA-2's performance for the Chinese corpus, while preserving its proficiency in English, critically hinges on two pivotal factors: the composition of the dataset, which encompasses both English and Chinese content, and the quality of each constituent dataset.

The following figure shows the data processing pipeline conducted for Colossal-LLaMA-2.
<p id="Colossal-LLaMA-2-data-processing-pipeline" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/data_processing_pipeline.jpeg?raw=true" width=800/>
</p>

❗️**Important**: We will open-source our data-processing toolkit soon, stay tuned!

### Tokenizer
The original LLaMA-2 vacabulary comprises fewer than a thousand Chinese characters, thus proves inadequate for encoding comprehensive Chinese texts effectively. Secondly, the utilization of byte tokens presents a challenge for transformer encoders to capture the semantic nuances of Chinese characters.

To address the above issues, we extend LLaMA-2 vocabulary from 32,000 to 69,104. To adapt the LLaMA-2 model for use with the Colossal-LLaMA-2 tokenizer, we initialize the new word embeddings by calculating the mean values from the original LLaMA-2 embeddings and subsequently append these new rows to the end of the original embedding matrices.

Advantages of extending vocabulary size:
* Improve the compression rate of string sequence encoding.
* Enhance the integrity of information.
* Enable encoded sequences to contain more valuable information, thereby theoretically enhancing the ability for chapter-level encoding.

Advantages of large vocabulary size under low-resource settings:
* The presence of numerous unused tokens can be attributed to the limited training dataset, where an excessive number of tokens might not have been effectively learned.
* Excessive vocabulary expansion leads to an increase in embedding-related parameters, resulting in higher memory usage, which, in turn, affects the efficiency of the training process.

To balance both sides, we finally construct our vocabulary with size 69,104. The following table below presents a comparison of various models at the 7B level.

| Model | Vocabulary Size | Compression Rate | Average Length of Samples (token-level) |
| :-----------: | :---------: | :----: | :----: |
| Colossal-LLaMA-2 | 69104 | 0.659 | 73.682 |
| LLaMA-2-7B | 32000 | 1.205 | 134.689 |
| Atom-7B | 65000 | 0.634 | 70.915 |
| Baichuan-7B | 64000 | 0.678 | 75.857 |
| Baichuan2-7B-base | 125696 | 0.570 | 63.761 |
| Chatglm2-6B | 64789 | 0.645 | 72.178 |
| InternLM-7B | 103168 | 0.566 | 63.349 |
| Qwen-7B | 151643 | 0.578 | 64.703 |
| Tigerbot-7B-base | 60515 | 0.630 | 70.515 |
| Yayi-7B-llama2 | 32005 | 1.214 | 135.689 |
| Chinese-llama-2-7b | 55296 | 0.668 | 74.690 |
| Chinese-Falcon-7B | 90046 | 0.669 | 74.858 |
| LinkSoul-Chinese-Llama-2-7b | 40076 | 0.958 | 107.089 |
| Ziya-LLaMA-13B-v1.1 | 39410 | 0.958 | 107.074 |


### Training Strategy
#### Multi-stage Training
In order to enhance the model's performance and harness the full potential of the original LLaMA-2, we have developed a multi-stage training strategy. This strategy is designed to systematically unlock the model's capabilities over a series of stages.

Therefore, we have divided the training process into three stages:
* Large-scale pre-training stage (Conducted by LLaMA-2): This initial stage is aimed at establishing the model's foundational capabilities from the ground up. It necessitates the use of a substantial dataset comprising no less than 1 trillion tokens.
* Chinese knowledge injection stage: In this stage, we introduce Chinese knowledge into the model. It requires access to a high-quality dataset rich in comprehensive knowledge relevant to the Chinese language.
* Knowledge replay stage: Knowledge is replayed through a question-answering (QA) mechanism, encompassing both the Chinese and English domains.

Following the completion of this multi-stage training process, the model exhibits notable improvements in performance across both English and Chinese benchmarks.

The following figure illustrates the three stages for training Colossal-LLaMA-2.

<p id="Colossal-LLaMA-2-Multi-stage-training" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/multi-stage-training.png?raw=true" width=600/>
</p>

#### Bucket-based Training
Our experiments have revealed that the distributions within the training dataset, as well as the arrangement of various topic-related data points, significantly impact the overall performance of the model, particularly in the context of continual pre-training of LLaMA-2.

In an effort to achieve a more balanced distribution and exert control over the dataset's ordering, we have adopted a method where we divide each sub-dataset into discrete bins. These bins are then combined to construct individual data buckets, with one bin contributed by each sub-dataset.

### Bridging Any Domain-specific Large Models
Applying the above process to perform knowledge transfer in any field allows for the cost-effective construction of lightweight domain-specific foundational large models.

<p id="domain_specific-llm" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/domain_specific-llm.jpeg?raw=true" width=800/>
</p>

## Citations
```bibtex
@article{bian2021colossal,
    title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
    author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
    journal={arXiv preprint arXiv:2110.14883},
    year={2021}
}
```
```bibtex
@misc{touvron2023llama,
    title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
    author={Hugo Touvron and Louis Martin and Kevin Stone and Peter Albert and Amjad Almahairi and Yasmine Babaei and Nikolay Bashlykov and Soumya Batra and Prajjwal Bhargava and Shruti Bhosale and Dan Bikel and Lukas Blecher and Cristian Canton Ferrer and Moya Chen and Guillem Cucurull and David Esiobu and Jude Fernandes and Jeremy Fu and Wenyin Fu and Brian Fuller and Cynthia Gao and Vedanuj Goswami and Naman Goyal and Anthony Hartshorn and Saghar Hosseini and Rui Hou and Hakan Inan and Marcin Kardas and Viktor Kerkez and Madian Khabsa and Isabel Kloumann and Artem Korenev and Punit Singh Koura and Marie-Anne Lachaux and Thibaut Lavril and Jenya Lee and Diana Liskovich and Yinghai Lu and Yuning Mao and Xavier Martinet and Todor Mihaylov and Pushkar Mishra and Igor Molybog and Yixin Nie and Andrew Poulton and Jeremy Reizenstein and Rashi Rungta and Kalyan Saladi and Alan Schelten and Ruan Silva and Eric Michael Smith and Ranjan Subramanian and Xiaoqing Ellen Tan and Binh Tang and Ross Taylor and Adina Williams and Jian Xiang Kuan and Puxin Xu and Zheng Yan and Iliyan Zarov and Yuchen Zhang and Angela Fan and Melanie Kambadur and Sharan Narang and Aurelien Rodriguez and Robert Stojnic and Sergey Edunov and Thomas Scialom},
    year={2023},
    eprint={2307.09288},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
```bibtex
@article{dao2023flashattention2,
    title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
    author={Dao, Tri},
    year={2023}
}
}
```
