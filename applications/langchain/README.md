# Colossalqa - A langchain based document retrieval conversation system
![Alt text](./examples/diagram.png?raw=true "Title")

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Install](#install)
- [How to Use](#how-to-use)
- Examples
  - Local Chinese Retrieval QA + Chat
    PATH/examples/retrieval_conversation_zh.py
  - Local English Retrieval QA + Chat
    PATH/examples/retrieval_conversation_en.py
  - Local Bi-lingual Retrieval QA + Chat
    PATH/examples/retrieval_conversation_universal.py
  - Experimental AI Agent Based on Chatgpt + Chat
    PATH/examples/conversation_agent_chatgpt.py

**As Colossal-AI is undergoing some major updates, this project will be actively maintained to stay in line with the Colossal-AI project.**

## Install

### Install the environment

```bash
conda create -n colossalqa
conda activate colossalqa
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI/applications/langchain
pip install -e .
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

## The Plan

- [x] build document retrieval QA tool
- [x] Add long + short term memory
- [x] Add demo for AI agent with SQL query
- [ ] Add customer retriever for fast construction and retrieving
- [ ] 
