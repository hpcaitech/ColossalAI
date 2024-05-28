import json
import os

from coati.dataset import setup_conversation_template
from coati.dataset.tokenization_utils import supervised_tokenize_sft
from transformers import AutoTokenizer

model_data_mapping = {
    "THUDM/chatglm2-6b": "THUDM_chatglm2-6b.json",
    "THUDM/chatglm3-6b": "THUDM_chatglm3-6b.json",
    "baichuan-inc/Baichuan2-13B-Chat": "baichuan-inc_Baichuan2-13B-Chat.json",
    "01-ai/Yi-1.5-9B-Chat": "01-ai_Yi-1.5-9B-Chat.json",
    "01-ai/Yi-34B": "01-ai_Yi-34B.json",
    "deepseek-ai/DeepSeek-V2-Lite": "deepseek-ai_DeepSeek-V2-Lite.json",
    "microsoft/phi-2": "microsoft_phi-2.json",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai_Mixtral-8x7B-Instruct-v0.1.json",
}
chat_template_config_path = "./config/conversation_template"


def test_tokenization_sft():
    for model in model_data_mapping:
        print(f"#############{model}#############")
        conversation_template_config = os.path.join(chat_template_config_path, model_data_mapping[model])
        messages = [
            {"from": "human", "content": "What are the three primary colors?"},
            {"from": "assistant", "content": "The three primary colors are red, blue, and yellow."},
            {"from": "human", "content": "解释个人电脑和服务器之间的区别。"},
            {
                "from": "assistant",
                "content": "个人电脑和服务器是两种不同类型的计算机系统，它们的主要区别在于用途、硬件配置和性能。 个人电脑，顾名思义，是为个人使用而设计的计算机。它们通常用于日常的工作、娱乐和学习，可以运行各种各样的应用程序和游戏。个人电脑的硬件配置一般是按照标准配置来设计的，不过也可以根据个人需求进行定制。 而服务器是为了满足大量用户的需求而设计的计算机系统，它们通常用于为用户提供各种网络服务，如网站、电子邮件和文件传输等。服务器通常需要高性能的硬件配置，并且可以承受高负载和长时间的运行。由于服务器需要支持大量用户的访问，它们通常配备多核处理器、大容量内存和大容量硬盘驱动器，以提高系统的运行速度和稳定性。 总之，个人电脑和服务器之间的主要区别在于它们的用途、硬件配置和性能。个人电脑用于个人使用，而服务器用于支持大量用户的访问。服务器的硬件配置通常比个人电脑更高，以保证系统的性能和稳定性。",
            },
        ]
        chat_template_config = json.load(open(conversation_template_config, "r", encoding="utf8"))
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
        conversation_template = setup_conversation_template(
            tokenizer, chat_template_config=chat_template_config, save_path=conversation_template_config
        )

        output = supervised_tokenize_sft({"messages": messages}, tokenizer, conversation_template)
        with open(f"./tests/test_data/chat_template/{model_data_mapping[model]}", "r", encoding="utf8") as f:
            assert json.dumps(json.load(f)) == json.dumps(output), f"model: {model} failed"
