# ColossalQA WebUI Demo

This demo provides a simple WebUI for ColossalQA, enabling you to upload your files as a knowledge base and interact with them through a chat interface in your browser.

The `server.py` initializes the backend RAG chain that can be backed by various language models (e.g., ChatGPT, Huawei Pangu, ChatGLM2). Meanwhile, `webui.py` launches a Gradio-supported chatbot interface.

# Usage

## Installation

First, install the necessary dependencies for ColossalQA:

```sh
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI/applications/ColossalQA/
pip install -e .
```

## Configure the RAG Chain

Customize the RAG Chain settings, such as the embedding model (default: moka-ai/m3e) and the language model, in the `start_colossal_qa.sh` script.

For API-based language models (like ChatGPT or Huawei Pangu), provide your API key for authentication. For locally-run models, indicate the path to the model's checkpoint file.

If you want to customize prompts in the RAG Chain, you can have a look at the `RAG_ChatBot.py` file to modify them.

## Run WebUI Demo

Execute the following command to start the demo:

```sh
bash start_colossal_qa.sh
```

After launching the script, you can upload files and engage with the chatbot through your web browser.

![ColossalQA Demo](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/colossalqa/img/qa_demo.png)