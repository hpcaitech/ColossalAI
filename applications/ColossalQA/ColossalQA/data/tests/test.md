# README Format File for Testing
![Alt text](./examples/diagram.png?raw=true "Fig.1. design of the document retrieval conversation system")

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Install](#install)
- [How to Use](#how-to-use)
- Examples
  - [Local Chinese Retrieval QA + Chat](examples/retrieval_conversation_zh.py)
  - [Local English Retrieval QA + Chat](examples/retrieval_conversation_en.py)
  - [Local Bi-lingual Retrieval QA + Chat](examples/retrieval_conversation_universal.py)
  - [Experimental AI Agent Based on Chatgpt + Chat](examples/conversation_agent_chatgpt.py)

**As Colossal-AI is undergoing some major updates, this project will be actively maintained to stay in line with the Colossal-AI project.**

## Install

Install colossalqa
```bash
# python==3.8.17
cd ColossalAI/applications/ColossalQA
pip install -e .
```

To use the vllm server, please refer to the official guide [here](https://github.com/vllm-project/vllm/tree/main) for installation instruction. Simply run the following command from another terminal.
```bash
cd ./vllm/entrypoints
python api_server.py --host localhost --port $PORT_NUMBER --model $PATH_TO_MODEL --swap-space $SWAP_SPACE_IN_GB
```

## How to use

### Collect your data

For ChatGPT based Agent we support document retrieval and simple sql search.
If you want to run the demo locally, we provided document retrieval based conversation system built upon langchain. It accept a wide range of documents. 

Read comments under ./colossalqa/data_loader for more detail 

### Serving
Currently use vllm will replace with colossal inference when ready. Please refer class VllmLLM.

### Run the script

We provided scripts for Chinese document retrieval based conversation system, English document retrieval based conversation system, Bi-lingual document retrieval based conversation system and an experimental AI agent with document retrieval and SQL query functionality.

To run the bi-lingual scripts, set the following environmental variables before running the script.
```bash
export ZH_MODEL_PATH=XXX
export ZH_MODEL_NAME: chatglm2
export EN_MODEL_PATH: XXX
export EN_MODEL_NAME: llama
python retrieval_conversation_universal.py
```

To run retrieval_conversation_en.py. set the following environmental variables.
```bash
export EN_MODEL_PATH=XXX
export EN_MODEL_NAME: llama
python retrieval_conversation_en.py
```

To run retrieval_conversation_zh.py. set the following environmental variables.
```bash
export ZH_MODEL_PATH=XXX
export ZH_MODEL_NAME: chatglm2
python retrieval_conversation_en.py
```

It will ask you to provide the path to your data during the execution of the script. You can also pass a glob path to load multiple files at once. If csv files are provided, please use ',' as delimiter and '"' as quotation mark. There are no other formatting constraints for loading documents type files. For loading table type files, we use pandas, please refer to [Pandas-Input/Output](https://pandas.pydata.org/pandas-docs/stable/reference/io.html) for file format details.

## The Plan

- [x] build document retrieval QA tool
- [x] Add long + short term memory
- [x] Add demo for AI agent with SQL query
- [x] Add customer retriever for fast construction and retrieving (with incremental mode)
