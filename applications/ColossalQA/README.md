# ColossalQA - Langchain-based Document Retrieval Conversation System

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overall Implementation](#overall-implementation)
- [Install](#install)
- [How to Use](#how-to-use)
- Examples
  - [Local Chinese Retrieval QA + Chat](examples/retrieval_conversation_zh.py)
  - [Local English Retrieval QA + Chat](examples/retrieval_conversation_en.py)
  - [Local Bi-lingual Retrieval QA + Chat](examples/retrieval_conversation_universal.py)
  - [Experimental AI Agent Based on Chatgpt + Chat](examples/conversation_agent_chatgpt.py)

**As Colossal-AI is undergoing some major updates, this project will be actively maintained to stay in line with the Colossal-AI project.**

## Overall Implementation

### Highlevel Design


![Alt text](./examples/diagram.png?raw=true "Fig.1. Design of the document retrieval conversation system")
<p align="center">
Fig.1. Design of the document retrieval conversation system
</p>

Retrieval-based Question Answering (QA) is a crucial application of natural language processing that aims to find the most relevant answers based on the information from a corpus of text documents in response to user queries. Vector stores, which represent documents and queries as vectors in a high-dimensional space, have gained popularity for their effectiveness in retrieval QA tasks.

#### Step 1: Collect Data

A successful retrieval QA system starts with high-quality data. You need a collection of text documents that's related to your application. You may also need to manually design how your data will be presented to the language model. 

#### Step 2: Split Data

Document data is usually too long to fit into the prompt due to the context length limitation of LLMs. Supporting documents need to be splited into short chunks before constructing vector stores. In this demo, we use neural text spliter for better performance.

#### Step 3: Construct Vector Stores
Choose a embedding function and embed your text chunk into high dimensional vectors. Once you have vectors for your documents, you need to create a vector store. The vector store should efficiently index and retrieve documents based on vector similarity. In this demo, we use [Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma) and incrementally update indexes of vector stores. Through incremental update, one can update and maintain a vector store without recalculating every embedding.
You are free to choose any vectorstore from a varity of [vector stores](https://python.langchain.com/docs/integrations/vectorstores/) supported by Langchain. However, the incremental update only works with LangChain vectorstore's that support:
- document addition by id (add_documents method with ids argument)
- delete by id (delete method with)

#### Step 4: Retrieve Relative Text 
Upon querying, we will run a reference resolution on user's input, the goal of this step is to remove ambiguous reference in user's query such as "this company", "him". We then embed the query with the same embedding function and query the vectorstore to retrieve the top-k most similar documents.

#### Step 5: Format Prompt
The prompt carries essential information including task description, conversation history, retrived documents, and user's query for the LLM to generate a response. Please refer to this [README](./colossalqa/prompt/README.md) for more details.

#### Step 6: Infer
Pass the prompt to the LLM with additional generaton arguments to get agent response. You can control the generation with additional arguments such as temperature, top_k, top_p, max_new_tokens. You can also define when to stop by passing the stop substring to the retrieval QA chain.

#### Step 7: Update Memory
We designed a memory module that incoporates both long-term memory and short-term memory. In this step, we update the memory with the newly generated response. To fix into the context length of a given LLM, we sumarize the overlength part of historical conversation and present the rest in round-based conversation format. Fig.2. shows how long-term and short-term memory are update. Please refer to this [README](./colossalqa/prompt/README.md) for dialogue format.

![Alt text](./examples/memory.png?raw=true "Fig.2. Design of the memory module")
<p align="center">
Fig.2. Design of the memory module
</p>

### Supported LLMs and Embedding Models
We support all language models that can be loaded by [```transformers.AutoModel.from_pretrained```](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#transformers.AutoModel.from_pretrained). However, since retrieval QA relies heavily on language model's zero-shot, instruction following and logic reasoning ability, small models are generally not recommended. In the local demo, we use ChatGLM2 for Chinese and LLaMa2 for English. To change the base LLM, you also need to modify the prompt accordingly. 

In this demo, we use ["moka-ai/m3e-base"](https://huggingface.co/moka-ai/m3e-base) as default embedding model. This model supports homogeneous text similarity calculation in both Chinese and English.

### Serving
Currently we provide an interface for infering with LLMs served by third party packages such as [vllm](https://github.com/vllm-project/vllm) we will replace it with colossal inference and serving when ready. Please refer class VllmLLM for more details.

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

## How to Use

### Collect Your Data

For ChatGPT based Agent we support document retrieval and simple sql search.
If you want to run the demo locally, we provided document retrieval based conversation system built upon langchain. It accept a wide range of documents. After collecting your data, put your data under a folder.

Read comments under ./colossalqa/data_loader for more detail regarding supported data formats.

### Run The Script

We provided scripts for Chinese document retrieval based conversation system, English document retrieval based conversation system, Bi-lingual document retrieval based conversation system and an experimental AI agent with document retrieval and SQL query functionality.

To run the bi-lingual scripts.
```bash
python retrieval_conversation_universal.py \
    --en_model_path /path/to/Llama-2-7b-hf \
    --zh_model_path /path/to/chatglm2-6b \
    --zh_model_name chatglm2 \
    --en_model_name llama \
    --sql_file_path /path/to/any/folder   
```

To run retrieval_conversation_en.py.
```bash
python retrieval_conversation_en.py \
    --data_path_en ../data/companies.txt \
    --en_model_path /path/to/Llama-2-7b-hf \
    --en_model_name llama \
    --sql_file_path /path/to/any/folder
```

To run retrieval_conversation_zh.py.
```bash
python retrieval_conversation_zh.py \
    --data_path_zh /data/scratch/test_data_colossalqa/companies_zh.txt \
    --zh_model_path /path/to/chatglm2-6b \
    --zh_model_name chatglm2 \
    --sql_file_path /path/to/any/folder
```

After runing the script, it will ask you to provide the path to your data during the execution of the script. You can also pass a glob path to load multiple files at once. Please read this [guide](https://docs.python.org/3/library/glob.html) on how to define glob path. Follow the instruction and provide all files for your retrieval conversation system then type "ESC" to finish loading documents. If csv files are provided, please use "," as delimiter and "\"" as quotation mark. For json and jsonl files. The default format is 
```
{
  "data":[
    {"content":"XXX"},
    {"content":"XXX"}
    ...
  ]
}
```

For other formats, please refer to [this document](https://python.langchain.com/docs/modules/data_connection/document_loaders/json) on how to define schema for data loading. There are no other formatting constraints for loading documents type files. For loading table type files, we use pandas, please refer to [Pandas-Input/Output](https://pandas.pydata.org/pandas-docs/stable/reference/io.html) for file format details.

## The Plan

- [x] build document retrieval QA tool
- [x] Add long + short term memory
- [x] Add demo for AI agent with SQL query
- [x] Add customer retriever for fast construction and retrieving (with incremental update)

## Reference
- [Langchain Repository: https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

- [Document Segmentation: https://modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base/quickstart](https://modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base/quickstart)

- [Incremental Update: https://python.langchain.com/docs/modules/data_connection/indexing?ref=blog.langchain.dev](https://python.langchain.com/docs/modules/data_connection/indexing?ref=blog.langchain.dev)

- [LangChain-Chatchat: https://github.com/chatchat-space/Langchain-Chatchat/tree/master](https://github.com/chatchat-space/Langchain-Chatchat/tree/master)

