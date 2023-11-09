import argparse
import copy
import json
import os
import random
import string
from http.server import BaseHTTPRequestHandler, HTTPServer
from colossalqa.local.llm import ColossalAPI, ColossalLLM
from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.retrieval_conversation_zh import ChineseRetrievalConversation
from colossalqa.retriever import CustomRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from RAG_ChatBot import RAG_ChatBot, DEFAULT_RAG_CFG

# Define the mapping between embed_model_name(passed from Front End) and the actual path on the back end server
EMBED_MODEL_DICT = {
    "m3e": os.environ.get("EMB_MODEL_PATH", DEFAULT_RAG_CFG["embed_model_name_or_path"])
}
# Define the mapping between LLM_name(passed from Front End) and the actual path on the back end server
LLM_DICT = {  
    "chatglm2": os.environ.get("CHAT_LLM_PATH", "THUDM/chatglm-6b"),
    "pangu": "Pangu_API",
    "chatgpt": "OpenAI_API"
}

def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length)) 

class ColossalQAServerRequestHandler(BaseHTTPRequestHandler):
    chatbot = None  
    def _set_response(self):
        """
        set http header for response
        """
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        received_json = json.loads(post_data.decode("utf-8"))
        print(received_json)
        # conversation_ready is False(user's first request): Need to upload files and initialize the RAG chain
        if received_json["conversation_ready"] is False: 
            self.rag_config = DEFAULT_RAG_CFG.copy()
            try:
                assert received_json["embed_model_name"] in EMBED_MODEL_DICT
                assert received_json["llm_name"] in LLM_DICT
                self.docs_files = received_json["docs"]
                embed_model_name, llm_name = received_json["embed_model_name"], received_json["llm_name"]
                
                # Find the embed_model/llm ckpt path on the back end server.
                embed_model_path, llm_path = EMBED_MODEL_DICT[embed_model_name], LLM_DICT[llm_name]  
                self.rag_config["embed_model_name_or_path"] = embed_model_path 

                # Create the storage path for knowledge base files
                self.rag_config["retri_kb_file_path"] = os.path.join(os.environ["TMP"], "colossalqa_kb/"+randomword(20))
                if not os.path.exists(self.rag_config["retri_kb_file_path"]):
                    os.makedirs(self.rag_config["retri_kb_file_path"])
                
                if (embed_model_path is not None) and (llm_path is not None):
                    # ---- Intialize LLM, QA_chatbot here ----
                    print("Initializing LLM...")
                    if llm_path == "Pangu_API":
                        from colossalqa.local.pangu_llm import Pangu
                        self.llm = Pangu(id=1)
                        self.llm.set_auth_config()  # verify user's auth info here
                        self.rag_config["mem_llm_kwargs"] = None
                        self.rag_config["disambig_llm_kwargs"] = None
                        self.rag_config["gen_llm_kwargs"] = None
                    elif llm_path == "OpenAI_API":
                        from langchain.llms import OpenAI
                        self.llm = OpenAI()
                        self.rag_config["mem_llm_kwargs"] = None
                        self.rag_config["disambig_llm_kwargs"] = None
                        self.rag_config["gen_llm_kwargs"] = None
                    else:
                        # ** (For Testing Only) **
                        # In practice, all LLMs will run on the cloud platform and accessed by API, instead of running locally.
                        # initialize model from model_path by using ColossalLLM 
                        self.rag_config["mem_llm_kwargs"] = {"max_new_tokens": 50, "temperature": 1, "do_sample": True}
                        self.rag_config["disambig_llm_kwargs"] = {"max_new_tokens": 30, "temperature": 1, "do_sample": True}
                        self.rag_config["gen_llm_kwargs"] = {"max_new_tokens": 100, "temperature": 1, "do_sample": True}
                        self.colossal_api = ColossalAPI(llm_name, llm_path)
                        self.llm = ColossalLLM(n=1, api=self.colossal_api)
                
                    print(f"Initializing RAG Chain...")
                    print("RAG_CONFIG: ", self.rag_config)
                    self.__class__.chatbot = RAG_ChatBot(self.llm, self.rag_config)
                    print("Loading Files....\n", self.docs_files)
                    self.__class__.chatbot.load_doc_from_files(self.docs_files)
                    # -----------------------------------------------------------------------------------
                    res = {"response": f"文件上传完成，模型初始化完成，让我们开始对话吧！(后端模型:{llm_name})", "error": "", "conversation_ready": True}
            except Exception as e:
                res = {"response": "文件上传或模型初始化有误，无法开始对话。",
                       "error": f"Error in File Uploading and/or RAG initialization. Error details: {e}", 
                       "conversation_ready": False}
        # conversation_ready is True: Chatbot and docs are all set. Ready to chat.
        else:  
            user_input = received_json["user_input"]
            chatbot_response, self.__class__.chatbot.memory = self.__class__.chatbot.run(user_input, self.__class__.chatbot.memory)
            res = {"response": chatbot_response, "error": "", "conversation_ready": True}
        self._set_response()
        self.wfile.write(json.dumps(res).encode("utf-8"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chinese retrieval based conversation system")
    parser.add_argument("--port", type=int, default=13666, help="port on localhost to start the server")
    args = parser.parse_args()
    server_address = ("localhost", args.port)
    httpd = HTTPServer(server_address, ColossalQAServerRequestHandler)
    print(f"Starting server on port {args.port}...")
    httpd.serve_forever()
    
