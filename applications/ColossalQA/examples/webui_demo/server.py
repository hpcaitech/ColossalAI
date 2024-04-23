import argparse
from typing import List, Union

import config
import uvicorn
from colossalqa.local.llm import ColossalAPI, ColossalLLM
from colossalqa.mylogging import get_logger
from fastapi import FastAPI, Request
from pydantic import BaseModel
from RAG_ChatBot import RAG_ChatBot
from utils import DocAction

logger = get_logger()


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http_host", default="0.0.0.0")
    parser.add_argument("--http_port", type=int, default=13666)
    return parser.parse_args()


app = FastAPI()


class DocUpdateReq(BaseModel):
    doc_files: Union[List[str], str, None] = None
    action: DocAction = DocAction.ADD


class GenerationTaskReq(BaseModel):
    user_input: str


@app.post("/update")
def update_docs(data: DocUpdateReq, request: Request):
    if data.action == "add":
        if isinstance(data.doc_files, str):
            data.doc_files = [data.doc_files]
        chatbot.load_doc_from_files(files=data.doc_files)
        all_docs = ""
        for doc in chatbot.docs_names:
            all_docs += f"\t{doc}\n\n"
        return {"response": f"文件上传完成，所有数据库文件：\n\n{all_docs}让我们开始对话吧！"}
    elif data.action == "clear":
        chatbot.clear_docs(**all_config["chain"])
        return {"response": f"已清空数据库。"}


@app.post("/generate")
def generate(data: GenerationTaskReq, request: Request):
    try:
        chatbot_response, chatbot.memory = chatbot.run(data.user_input, chatbot.memory)
        return {"response": chatbot_response, "error": ""}
    except Exception as e:
        return {"response": "模型生成回答有误", "error": f"Error in generating answers, details: {e}"}


if __name__ == "__main__":
    args = parseArgs()

    all_config = config.ALL_CONFIG
    model_name = all_config["model"]["model_name"]

    # initialize chatbot
    logger.info(f"Initialize the chatbot from {model_name}")

    if all_config["model"]["mode"] == "local":
        colossal_api = ColossalAPI(model_name, all_config["model"]["model_path"])
        llm = ColossalLLM(n=1, api=colossal_api)
    elif all_config["model"]["mode"] == "api":
        if model_name == "pangu_api":
            from colossalqa.local.pangu_llm import Pangu

            gen_config = {
                "user": "User",
                "max_tokens": all_config["chain"]["disambig_llm_kwargs"]["max_new_tokens"],
                "temperature": all_config["chain"]["disambig_llm_kwargs"]["temperature"],
                "n": 1,  # the number of responses generated
            }
            llm = Pangu(gen_config=gen_config)
            llm.set_auth_config()  # verify user's auth info here
        elif model_name == "chatgpt_api":
            from langchain.llms import OpenAI

            llm = OpenAI()
    else:
        raise ValueError("Unsupported mode.")

    # initialize chatbot
    chatbot = RAG_ChatBot(llm, all_config)

    app_config = uvicorn.Config(app, host=args.http_host, port=args.http_port)
    server = uvicorn.Server(config=app_config)
    server.run()
