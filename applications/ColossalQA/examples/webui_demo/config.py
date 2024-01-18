from colossalqa.prompt.prompt import (
    PROMPT_DISAMBIGUATE_ZH,
    PROMPT_RETRIEVAL_QA_ZH,
    SUMMARY_PROMPT_ZH,
    ZH_RETRIEVAL_QA_REJECTION_ANSWER,
    ZH_RETRIEVAL_QA_TRIGGER_KEYWORDS,
)
from colossalqa.text_splitter import ChineseTextSplitter

ALL_CONFIG = {
    "embed": {
        "embed_name": "m3e",  # embedding model name
        "embed_model_name_or_path": "moka-ai/m3e-base",  # path to embedding model, could be a local path or a huggingface path
        "embed_model_device": {
            "device": "cpu"
        }
    },
    "model": {
        "mode": "api",  # "local" for loading models, "api" for using model api
        "model_name": "chatgpt_api",  # local model name, "chatgpt_api" or "pangu_api"
        "model_path": "", # path to the model, could be a local path or a huggingface path. don't need if using an api
        "device": {
            "device": "cuda"
        }
    },
    "splitter": {
        "name": ChineseTextSplitter
    },
    "retrieval": {
        "retri_top_k": 3,
        "retri_kb_file_path": "./", # path to store database files
        "verbose": True
    },
    "chain": {
        "mem_summary_prompt": SUMMARY_PROMPT_ZH,  # summary prompt template
        "mem_human_prefix": "用户",
        "mem_ai_prefix": "Assistant",
        "mem_max_tokens": 2000,
        "mem_llm_kwargs": {
            "max_new_tokens": 50,
            "temperature": 1,
            "do_sample": True
        },
        "disambig_prompt": PROMPT_DISAMBIGUATE_ZH,  # disambiguate prompt template
        "disambig_llm_kwargs": {
            "max_new_tokens": 30,
            "temperature": 1,
            "do_sample": True
        },
        "gen_llm_kwargs": {
            "max_new_tokens": 100,
            "temperature": 1,
            "do_sample": True
        },
        "gen_qa_prompt": PROMPT_RETRIEVAL_QA_ZH,  # generation prompt template
        "verbose": True    
    }   
}