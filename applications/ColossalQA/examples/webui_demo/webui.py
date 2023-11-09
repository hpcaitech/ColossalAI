import json
import os
import gradio as gr
import requests

RAG_STATE = {"conversation_ready": False,  # Conversation is not ready until files are uploaded and RAG chain is initialized
             "embed_model_name": os.environ.get("EMB_MODEL", "m3e"),
             "llm_name": os.environ.get("CHAT_LLM", "chatgpt")}  
URL = "http://localhost:13666"

def get_response(client_data, URL):
    headers = {"Content-type": "application/json"}
    print(f"Sending request to server url: {URL}")
    response = requests.post(URL, data=json.dumps(client_data), headers=headers)
    response = json.loads(response.content)
    return response

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value=None, interactive=True)

def add_file(history, files):
    global RAG_STATE
    RAG_STATE["conversation_ready"] = False  # after adding new files, reset the ChatBot
    RAG_STATE["upload_files"]=[file.name for file in files]
    files_string = "\n".join([os.path.basename(path) for path in RAG_STATE["upload_files"]])
    print(files_string)
    history = history + [(files_string, None)]
    return history

def bot(history):
    print(history)
    global RAG_STATE
    if not RAG_STATE["conversation_ready"]:
        # Upload files and initialize models
        client_data = {
            "docs": RAG_STATE["upload_files"],
            "embed_model_name": RAG_STATE["embed_model_name"],  # Select embedding model name here
            "llm_name": RAG_STATE["llm_name"],  # Select LLM model name here. ["pangu", "chatglm2"]
            "conversation_ready": RAG_STATE["conversation_ready"]
        }
    else:
        client_data = {}
        client_data["conversation_ready"] = RAG_STATE["conversation_ready"]
        client_data["user_input"] = history[-1][0].strip()

    response = get_response(client_data, URL)  # TODO: async request, to avoid users waiting the model initialization too long
    print(response)
    if response["error"] != "":
        raise gr.Error(response["error"])
    
    RAG_STATE["conversation_ready"] = response["conversation_ready"]
    history[-1][1] = response["response"]
    yield history


CSS = """
.contain { display: flex; flex-direction: column; height: 100vh }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; }
"""

header_html = """
<div style="background: linear-gradient(to right, #2a0cf4, #7100ed, #9800e6, #b600df, #ce00d9, #dc0cd1, #e81bca, #f229c3, #f738ba, #f946b2, #fb53ab, #fb5fa5); padding: 20px; text-align: left;">
    <h1 style="color: white;">ColossalQA</h1>
    <h4 style="color: white;">ColossalQA</h4>
</div>
"""

with gr.Blocks(css=CSS) as demo:
    html = gr.HTML(header_html)
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(
            (os.path.join(os.path.dirname(__file__), "img/avatar_user.png")),
            (os.path.join(os.path.dirname(__file__), "img/avatar_ai.png")),
        ),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=True,
            autofocus=True,
        )
        btn = gr.UploadButton("📁", file_types=["file"], file_count="multiple")

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(bot, chatbot, chatbot)
    # Clear the original textbox
    txt_msg.then(lambda: gr.update(value=None, interactive=True), None, [txt], queue=False) 
    # Click Upload Button: 1. upload files  2. send config to backend, initalize model 3. get response "conversation_ready" = True/False
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(bot, chatbot, chatbot)



if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)  # share=True will release a public link of the demo
