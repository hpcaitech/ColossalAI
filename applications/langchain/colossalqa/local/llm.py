import argparse

import torch
from transformers import AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer, LlamaTokenizer, AutoModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Optional, List, Dict, Mapping, Any
from .utils import post_http_request, get_response
from coati.models.bloom import BLOOMActor
from coati.models.gpt import GPTActor
from coati.models.llama import LlamaActor
from coati.models.opt import OPTActor

class CoatiAPI:
    def __init__(self, model_type: str, pretrain: str, ckpt_path: str=None) -> None:
        # configure model'
        self.model_type = model_type
        if model_type == 'gpt2':
            self.actor = GPTActor(pretrained=pretrain)
        elif model_type == 'bloom':
            self.actor = BLOOMActor(pretrained=pretrain)
        elif model_type == 'opt':
            self.actor = OPTActor(pretrained=pretrain)
        elif model_type == 'llama':
            self.actor = LlamaActor(pretrained=pretrain)
            # self.actor = LlamaForCausalLM.from_pretrained(pretrain, torch_dtype=torch.float16, trust_remote_code=True)
        elif model_type == 'chatglm' or model_type == 'chatglm2':
            self.actor = AutoModel.from_pretrained(pretrain, torch_dtype=torch.float16, trust_remote_code=True)
        else:
            raise ValueError(f'Unsupported model "{model_type}"')

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)
            self.actor.load_state_dict(state_dict)
        self.actor.to(torch.cuda.current_device())

        # configure tokenizer
        if model_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif model_type == 'bloom':
            self.tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif model_type == 'opt':
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif model_type == 'llama':
            self.tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
            self.tokenizer.eos_token = '<\s>'
            self.tokenizer.pad_token = self.tokenizer.unk_token
        elif model_type == "chatglm":
            self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        elif model_type == "chatglm2":
            self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        else:
            raise ValueError(f'Unsupported model "{model_type}"')

        self.actor.eval()

    def generate(self, input: str, **kwargs):    
        if self.model_type in ['chatglm', 'chatglm2']:
            inputs = {k: v.to(torch.cuda.current_device()) for k, v in self.tokenizer(input, return_tensors="pt").items()}
            output = self.actor.generate(**inputs, **kwargs)
            output = output.cpu()
            prompt_len = inputs['input_ids'].size(1)
            response = output[0, prompt_len:]
            output = self.tokenizer.decode(response, skip_special_tokens=True)
            return output
        else:
            inputs = {'input_ids': self.tokenizer(input, return_tensors="pt")['input_ids'].to(torch.cuda.current_device())}
            # kwargs['max_length'] = kwargs['max_new_tokens'] + input_ids.size()[-1]
            outputs = self.actor.model.generate(
                            **inputs,
                            **kwargs)
            output = outputs.cpu()
            prompt_len = inputs['input_ids'].size(1)
            response = output[0, prompt_len:]
            output = self.tokenizer.decode(response, skip_special_tokens=True)
            return output
                
    def get_embedding(self, input: str):
        pass

class VllmAPI:
    def __init__(self, host:str='localhost', port:int=8077) -> None:
        # configure model
        self.host = host
        self.port = port
        self.url =  f"http://{self.host}:{self.port}/generate"

    def generate(self, input: str, **kwargs):    
        output = get_response(post_http_request(input, self.url, **kwargs))[0]
        return output[len(input):]
    
    def get_embedding(self, input: str):
        pass

# langchain LLM wrapper
class CoatiLLM(LLM):
    n: int
    api: Any
    kwargs = {}

    @property
    def _llm_type(self) -> str:
        return "custom"
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        print(f"kwargs:{kwargs}, prompt:{prompt}, stop:{stop}")
        for k in self.kwargs:
            if k not in kwargs:
                kwargs[k] = self.kwargs[k]

        generate_args = {k:kwargs[k] for k in kwargs if k not in ['stop', 'n']}
        out = self.api.generate(prompt, **generate_args)
        print("---------------------------")
        if len(stop)!=0:
            for stopping_words in stop:
                if stopping_words in out:
                    out = out.split(stopping_words)[0]
        print(out)
        return out
    
    def set_api(self, api: Any, **kwargs) -> None:
        if 'max_new_tokens' not in kwargs:
            kwargs['max_new_tokens']=100
        self.kwargs = kwargs
        self.api = api
        print("done")

    def set_kwargs(self, **kwargs) -> None:
        print(f"set kwargs:{kwargs}")
        if 'max_new_tokens' not in kwargs:
            kwargs['max_new_tokens']=100
        self.kwargs = kwargs

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
    
# langchain LLM wrapper
class VllmLLM(LLM):
    n: int
    api: Any
    kwargs = {}

    @property
    def _llm_type(self) -> str:
        return "custom"
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        # print(kwargs)
        for k in self.kwargs:
            if k not in kwargs:
                kwargs[k] = self.kwargs[k]
        # print(prompt)
        generate_args = {k:kwargs[k] for k in kwargs if k in ['n','max_tokens','temperature','stream']}
        out = self.api.generate(prompt, **generate_args)
        # print("---------------------------")
        if len(stop)!=0:
            for stopping_words in stop:
                if stopping_words in out:
                    out = out.split(stopping_words)[0]
        # print(out)
        return out
    
    def set_host_port(self, host: str='localhost', port: int=8077, **kwargs) -> None:
        self.kwargs = kwargs
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens']=100
        self.api = VllmAPI(host=host, port=port)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
    
    
if __name__ == '__main__':
    import os
    llm = CoatiLLM(n=10)
    model_path = os.environ.get('CHATGLM_PATH')
    llm.set_llm("chatglm", model_path)
    template = '''Have a conversation with a human, answering the following questions as best you can. You have access to the following tools. If none of the tools available can be applied to answer the question or you think answering the question doesn't need any additional tool, try to answer the question directly

Query the SQL database regarding /home/lcyab/data3/langchain/langchain/data/test_csv_organization_100.csv: useful for when you need to answer questions based on data stored on a SQL database regarding /home/lcyab/data3/langchain/langchain/data/test_csv_organization_100.csv

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Query the SQL database regarding /home/lcyab/data3/langchain/langchain/data/test_csv_organization_100.csv, Answer directly]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Please follow the output format of the following examples.
Example 1:
Question: Which companies are located in China
Output:
Thought: I should use a tool that will allow me to do calculation
Action: Calculate
Action Input: 1 + 1

Example 2: 
Question: Hi
Output:
Thought: How do I greet someone?
Action: None needed.
Final Answer: Hi!

You are provided with the following background knowledge:
Begin!"
Historical conversation:

Supporting materials:


Question: Hi
Output:
'''
    print(llm("Question: what dog doesn't bark? Answer:", max_new_tokens=10))