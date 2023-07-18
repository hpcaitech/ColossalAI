# Data PreProcessing for chinese Whole Word Masked

<span id='all_catelogue'/>

## Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#Quick Start Guide'>2. Quick Start Guide:</a>
    * <a href='#Split Sentence'>2.1. Split Sentence</a>
    * <a href='#Tokenizer & Whole Word Masked'>2.2.Tokenizer & Whole Word Masked</a>


<span id='introduction'/>

## 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>
This folder is used to preprocess chinese corpus with Whole Word Masked. You can obtain corpus from [WuDao](https://resource.wudaoai.cn/home?ind&name=WuDaoCorpora%202.0&id=1394901288847716352). Moreover, data preprocessing is flexible, and you can modify the code based on your needs, hardware or parallel framework(Open MPI, Spark, Dask).

<span id='Quick Start Guide'/>

## 2. Quick Start Guide: <a href='#all_catelogue'>[Back to Top]</a>

<span id='Split Sentence'/>

### 2.1. Split Sentence & Split data into multiple shard:
Firstly, each file has multiple documents, and each document contains multiple sentences. Split sentence through punctuation, such as `。！`. **Secondly, split data into multiple shard based on server hardware (cpu, cpu memory, hard disk) and corpus size.** Each shard contains a part of corpus, and the model needs to train all the shards as one epoch.
In this example, split 200G Corpus into 100 shard, and each shard is about 2G. The size of the shard is memory-dependent, taking into account the number of servers, the memory used by the tokenizer, and the memory used by the multi-process training to read the shard (n data parallel requires n\*shard_size memory). **To sum up, data preprocessing and model pretraining requires fighting with hardware, not just GPU.**

```python
python sentence_split.py --input_path /original_corpus --output_path /shard --shard 100
# This step takes a short time
```
* `--input_path`: all original corpus, e.g., /original_corpus/0.json /original_corpus/1.json ...
* `--output_path`: all shard with split sentences, e.g., /shard/0.txt, /shard/1.txt ...
* `--shard`: Number of shard, e.g., 10, 50, or 100

<summary><b>Input json:</b></summary>

```
[
    {
        "id": 0,
        "title": "打篮球",
        "content": "我今天去打篮球。不回来吃饭。"
    }
    {
        "id": 1,
        "title": "旅游",
        "content": "我后天去旅游。下周请假。"
    }
]
```

<summary><b>Output txt:</b></summary>

```
我今天去打篮球。
不回来吃饭。
]]
我后天去旅游。
下周请假。
```

<span id='Tokenizer & Whole Word Masked'/>

### 2.2. Tokenizer & Whole Word Masked:

```python
python tokenize_mask.py --input_path /shard --output_path /h5 --tokenizer_path /roberta --backend python
# This step is time consuming and is mainly spent on mask
```

**[optional but recommended]**: the C++ backend with `pybind11` can provide faster speed

```shell
make
```

* `--input_path`: location of all shard with split sentences, e.g., /shard/0.txt, /shard/1.txt ...
* `--output_path`: location of all h5 with token_id, input_mask, segment_ids and masked_lm_positions, e.g., /h5/0.h5, /h5/1.h5 ...
* `--tokenizer_path`: tokenizer path contains huggingface tokenizer.json. Download config.json, special_tokens_map.json, vocab.txt and tokenizer.json from [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/tree/main)
* `--backend`: python or c++, **specifies c++ can obtain faster preprocess speed**
* `--dupe_factor`: specifies how many times the preprocessor repeats to create the input from the same article/document
* `--worker`: number of process

<summary><b>Input txt:</b></summary>

```
我今天去打篮球。
不回来吃饭。
]]
我后天去旅游。
下周请假。
```

<summary><b>Output h5+numpy:</b></summary>

```
'input_ids': [[id0,id1,id2,id3,id4,id5,id6,0,0..],
              ...]
'input_mask': [[1,1,1,1,1,1,0,0..],
               ...]
'segment_ids': [[0,0,0,0,0,...],
               ...]
'masked_lm_positions': [[label1,-1,-1,label2,-1...],
                        ...]
```
