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

Customize the RAG Chain settings, such as the embedding model (default: moka-ai/m3e), the language model, and the prompts, in the `config.py`.

For API-based language models (like ChatGPT or Huawei Pangu), provide your API key for authentication. For locally-run models, indicate the path to the model's checkpoint file.

## Run WebUI Demo

Execute the following command to start the demo:

1. If you want to use a local model as the backend model, you need to specify the model name and model path in `config.py` and run the following commands.

```sh
export TMP="path/to/store/tmp/files"
# start the backend server
python server.py --http_host "host" --http_port "port"

# in an another terminal, start the ui
python webui.py --http_host "your-backend-api-host" --http_port "your-backend-api-port"
```

2. If you want to use pangu api as the backend model, you need to change the model mode to "api", change the model name to "chatgpt_api" in `config.py`, and run the following commands.
```sh
export TMP="path/to/store/tmp/files"

# Auth info for OpenAI API
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# start the backend server
python server.py --http_host "host" --http_port "port"

# in an another terminal, start the ui
python webui.py --http_host "your-backend-api-host" --http_port "your-backend-api-port"
```

3. If you want to use pangu api as the backend model, you need to change the model mode to "api", change the model name to "pangu_api" in `config.py`, and run the following commands.
```sh
export TMP="path/to/store/tmp/files"

# Auth info for Pangu API
export URL=""
export USERNAME=""
export PASSWORD=""
export DOMAIN_NAME=""

# start the backend server
python server.py --http_host "host" --http_port "port"

# in an another terminal, start the ui
python webui.py --http_host "your-backend-api-host" --http_port "your-backend-api-port"
```

After launching the script, you can upload files and engage with the chatbot through your web browser.

![ColossalQA Demo](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/colossalqa/new_ui.png)