import argparse
import json
import os

import gradio as gr
import requests
from utils import DocAction


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http_host", default="0.0.0.0")
    parser.add_argument("--http_port", type=int, default=13666)
    return parser.parse_args()


def get_response(data, url):
    headers = {"Content-type": "application/json"}
    response = requests.post(url, json=data, headers=headers)
    response = json.loads(response.content)
    return response


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value=None, interactive=True)


def add_file(history, files):
    files_string = "\n".join([os.path.basename(file.name) for file in files])

    doc_files = [file.name for file in files]
    data = {"doc_files": doc_files, "action": DocAction.ADD}
    response = get_response(data, update_url)["response"]
    history = history + [(files_string, response)]
    return history


def bot(history):
    data = {"user_input": history[-1][0].strip()}
    response = get_response(data, gen_url)

    if response["error"] != "":
        raise gr.Error(response["error"])

    history[-1][1] = response["response"]
    yield history


def restart(chatbot, txt):
    # Reset the conversation state and clear the chat history
    data = {"doc_files": "", "action": DocAction.CLEAR}
    get_response(data, update_url)

    return gr.update(value=None), gr.update(value=None, interactive=True)


CSS = """
.contain { display: flex; flex-direction: column; height: 100vh }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; }
"""

header_html = """
<div style="background: linear-gradient(to right, #2a0cf4, #7100ed, #9800e6, #b600df, #ce00d9, #dc0cd1, #e81bca, #f229c3, #f738ba, #f946b2, #fb53ab, #fb5fa5); padding: 20px; text-align: left;">
    <h1 style="color: white;">ColossalQA</h1>
    <h4 style="color: white;">A powerful Q&A system with knowledge bases</h4>
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
        btn = gr.UploadButton("📁", file_types=["file"], file_count="multiple", size="sm")
        restart_btn = gr.Button(str("\u21BB"), elem_id="restart-btn", scale=1)
        txt = gr.Textbox(
            scale=8,
            show_label=False,
            placeholder="Enter text and press enter, or use 📁 to upload files, click \u21BB to clear loaded files and restart chat",
            container=True,
            autofocus=True,
        )

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(bot, chatbot, chatbot)
    # Clear the original textbox
    txt_msg.then(lambda: gr.update(value=None, interactive=True), None, [txt], queue=False)
    # Click Upload Button: 1. upload files  2. send config to backend, initalize model 3. get response "conversation_ready" = True/False
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False)

    # restart
    restart_msg = restart_btn.click(restart, [chatbot, txt], [chatbot, txt], queue=False)


if __name__ == "__main__":
    args = parseArgs()

    update_url = f"http://{args.http_host}:{args.http_port}/update"
    gen_url = f"http://{args.http_host}:{args.http_port}/generate"

    demo.queue()
    demo.launch(share=True)  # share=True will release a public link of the demo
