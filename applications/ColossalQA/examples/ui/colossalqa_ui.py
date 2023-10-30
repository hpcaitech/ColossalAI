import json
import os

import gradio as gr
import requests

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

client_data = {
    "docs": [],
    "embedding": "moka-ai/m3e-base",
    "model_path": "/mnt/vepfs/lcxyc/leaderboard_models/chatglm2-6b",  # Define model path here
    "model_name": "chatglm2",  # Define model name here
    "sql_file_path": None,
    "user_input": None,
}
URL = "http://localhost:13666"


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def bot(history):
    print(history)
    global client_data
    assert client_data["model_name"] is not None, "Please define model name"
    assert client_data["model_path"] is not None, "Please define model path"
    assert client_data["embedding"] is not None, "Please define embedding model"
    if not isinstance(history[-1][0], str):
        client_data["docs"] = list(history[-1][0])
        client_data["user_input"] = None
    else:
        client_data["user_input"] = history[-1][0]
    if client_data["user_input"] is not None:
        client_data["user_input"] = client_data["user_input"].strip()
        headers = {"Content-type": "application/json"}
        print(f"Sending request to server url: {URL}")
        response = requests.post(URL, data=json.dumps(client_data), headers=headers)
        response = json.loads(response.text)
        history[-1][1] = response["response"]
        if response["error"] != "":
            raise gr.Error(response["error"])
        client_data["sql_file_path"] = response["sql_file_path"]
    else:
        history[-1][1] = "---文件已上传---"

    yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(
            (os.path.join(os.path.dirname(__file__), "img/avatar_ai.png")),
            (os.path.join(os.path.dirname(__file__), "img/avatar_user.png")),
        ),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(bot, chatbot, chatbot)
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(bot, chatbot, chatbot)

demo.queue()
if __name__ == "__main__":
    demo.launch(share=True)
