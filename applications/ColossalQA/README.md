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
```
# Install vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .  # This may take 5-10 minutes.
```
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

We provided scripts for Chinese document retrieval based conversation system, English document retrieval based conversation system, Bi-lingual document retrieval based conversation system and an experimental AI agent with document retrieval and SQL query functionality. The Bi-lingual one is a high-level wrapper for the other two clases. We write different scripts for different languages because retrieval QA requires different embedding models, LLMs, prompts for different language setting. For now, we use LLaMa2 for English retrieval QA and ChatGLM2 for Chinese retrieval QA for better performance.

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
    --model_path /path/to/Llama-2-7b-hf \
    --model_name llama \
    --sql_file_path /path/to/any/folder
```

To run retrieval_conversation_zh.py.
```bash
python retrieval_conversation_zh.py \
    --model_path /path/to/chatglm2-6b \
    --model_name chatglm2 \
    --sql_file_path /path/to/any/folder
```

To run retrieval_conversation_chatgpt.py.
```bash
python retrieval_conversation_chatgpt.py \
    --open_ai_key_path /path/to/plain/text/openai/key/file \
    --sql_file_path /path/to/any/folder
```

To run conversation_agent_chatgpt.py.
```bash
python conversation_agent_chatgpt.py \
    --open_ai_key_path /path/to/plain/text/openai/key/file
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

We also support another kay-value mode that utilizes a user-defined key to calculate the embeddings of the vector store. If a query matches a specific key, the value corresponding to that key will be used to generate the prompt. For instance, in the document below, "My coupon isn't working." will be employed during indexing, whereas "Question: My coupon isn't working.\nAnswer: We apologize for ... apply it to?" will appear in the final prompt. This format is typically useful when the task involves carrying on a conversation with readily accessible conversation data, such as customer service, question answering.
```python
Document(page_content="My coupon isn't working.", metadata={'is_key_value_mapping': True, 'seq_num': 36, 'source': 'XXX.json', 'value': "Question: My coupon isn't working.\nAnswer:We apologize for the inconvenience. Can you please provide the coupon code and the product name or SKU you're trying to apply it to?"})
```

For now, we only support the key-value mode for json data files. You can run the script retrieval_conversation_en_customer_service.py by the following command.

```bash
python retrieval_conversation_en_customer_service.py \
    --model_path /path/to/Llama-2-7b-hf \
    --model_name llama \
    --sql_file_path /path/to/any/folder
```

## The Plan

- [x] build document retrieval QA tool
- [x] Add long + short term memory
- [x] Add demo for AI agent with SQL query
- [x] Add customer retriever for fast construction and retrieving (with incremental update)

## Reference

```bibtex
@software{Chase_LangChain_2022,
author = {Chase, Harrison},
month = oct,
title = {{LangChain}},
url = {https://github.com/hwchase17/langchain},
year = {2022}
}
```
```bibtex
@inproceedings{DBLP:conf/asru/ZhangCLLW21,
  author    = {Qinglin Zhang and
               Qian Chen and
               Yali Li and
               Jiaqing Liu and
               Wen Wang},
  title     = {Sequence Model with Self-Adaptive Sliding Window for Efficient Spoken
               Document Segmentation},
  booktitle = {{IEEE} Automatic Speech Recognition and Understanding Workshop, {ASRU}
               2021, Cartagena, Colombia, December 13-17, 2021},
  pages     = {411--418},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://doi.org/10.1109/ASRU51503.2021.9688078},
  doi       = {10.1109/ASRU51503.2021.9688078},
  timestamp = {Wed, 09 Feb 2022 09:03:04 +0100},
  biburl    = {https://dblp.org/rec/conf/asru/ZhangCLLW21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
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
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```
```bibtex
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```
