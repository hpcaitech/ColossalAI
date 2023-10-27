import argparse
import copy
import json
import random
import string
from http.server import BaseHTTPRequestHandler, HTTPServer

from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.retrieval_conversation_zh import ChineseRetrievalConversation
from colossalqa.retriever import CustomRetriever
from colossalqa.text_splitter import ChineseTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


class ColossalQAServerRequestHandler(BaseHTTPRequestHandler):
    def chatbot_call(self, args) -> None:
        if args["sql_file_path"] is None:
            args["sql_file_path"] = "./" + randomword(20)
        if args["docs"] == []:
            return {"response": "", "error": "No documents provided", "sql_file_path": args["sql_file_path"]}
        if (
            "docs" not in self.__dict__
            or str(args["docs"]) != str(self.docs)
            or args["embedding"] != self.embedding_model
        ):
            if str(args["embedding"]) != str(args["embedding"]):
                self.embedding = HuggingFaceEmbeddings(
                    model_name=args["embedding"],
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": False},
                )
                self.embedding_model = args["embedding"]
            self.docs = copy.deepcopy(args["docs"])
            self.information_retriever = CustomRetriever(k=3, sql_file_path=args["sql_file_path"], verbose=True)
            retriever_data = DocumentLoader(args["docs"]["files"]).all_data
            # Split
            text_splitter = ChineseTextSplitter()
            # print(retriever_data)
            documents = text_splitter.split_documents(retriever_data)
            self.information_retriever.add_documents(
                docs=documents, cleanup="incremental", mode="by_source", embedding=self.embedding
            )

            self.chinese_retrieval_conversation = ChineseRetrievalConversation.from_retriever(
                self.information_retriever_zh, model_path=args["model_path"], model_name=args["model_name"]
            )
        answer, _ = self.chinese_retrieval_conversation.run(args["user_input"], None)
        return {"response": answer, "error": "", "sql_file_path": args["sql_file_path"]}

    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_POST(self, post_data):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        print(post_data)
        received_json = json.loads(post_data.decode("utf-8"))

        res = self.chatbot_call(received_json)
        self._set_response()
        self.wfile.write(json.dumps(res).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chinese retrieval based conversation system backed by ChatGLM2")
    parser.add_argument("--port", type=int, default=50000, help="port on localhost to start the server")

    args = parser.parse_args()
    server_address = ("127.0.0.1", args.port)
    httpd = HTTPServer(server_address, ColossalQAServerRequestHandler)
    print(f"Starting server on port {args.port}...")
    httpd.serve_forever()
