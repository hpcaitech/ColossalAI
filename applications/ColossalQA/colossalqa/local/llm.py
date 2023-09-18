'''
API and LLM warpper class for running LLMs locally
'''
from typing import Optional, List, Mapping, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    LlamaForCausalLM,
    LlamaTokenizer
)
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from colossalqa.mylogging import get_logger
from .utils import post_http_request, get_response

logger = get_logger()

class ColossalAPI:
    '''
    API for calling LLM.generate
    '''
    def __init__(self, model_type: str, pretrain: str, ckpt_path: str=None) -> None:
        '''
        Configurate model
        '''
        self.model_type = model_type
        if model_type == 'llama':
            self.actor = LlamaForCausalLM.from_pretrained(pretrain, torch_dtype=torch.float16, trust_remote_code=True)
        else:
            self.actor = AutoModel.from_pretrained(pretrain, torch_dtype=torch.float16, trust_remote_code=True)
    
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)
            self.actor.load_state_dict(state_dict)
        self.actor.to(torch.cuda.current_device())

        # Configurate tokenizer
        if model_type == 'llama':
            self.tokenizer = LlamaTokenizer.from_pretrained(pretrain)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)
        
        self.actor.eval()

    def generate(self, input: str, **kwargs)->str: 
        '''
        Generate response given the prompt
        Args:
            input: input string
            **kwargs: language model keyword type arguments, such as top_k, top_p, temperature, max_new_tokens...
        Returns:
            output: output string
        '''   
        if self.model_type in ['chatglm', 'chatglm2']:
            inputs = {k: v.to(torch.cuda.current_device()) for k, v in self.tokenizer(input, return_tensors="pt").items()}
        else:
            inputs = {'input_ids': self.tokenizer(input, return_tensors="pt")['input_ids'].to(torch.cuda.current_device())}
        output = self.actor.generate(**inputs, **kwargs)
        output = output.cpu()
        prompt_len = inputs['input_ids'].size(1)
        response = output[0, prompt_len:]
        output = self.tokenizer.decode(response, skip_special_tokens=True)
        return output
    

class VllmAPI:
    def __init__(self, host:str='localhost', port:int=8077) -> None:
        # Configurate api for model served through web
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}/generate"

    def generate(self, input: str, **kwargs):    
        output = get_response(post_http_request(input, self.url, **kwargs))[0]
        return output[len(input):]
    


class ColossalLLM(LLM):
    """
    Langchain LLM wrapper for a local LLM
    """
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
        logger.info(f"kwargs:{kwargs}\nstop:{stop}\nprompt:{prompt}", verbose=self.verbose)
        for k in self.kwargs:
            if k not in kwargs:
                kwargs[k] = self.kwargs[k]

        generate_args = {k:kwargs[k] for k in kwargs if k not in ['stop', 'n']}
        out = self.api.generate(prompt, **generate_args)
        if len(stop)!=0:
            for stopping_words in stop:
                if stopping_words in out:
                    out = out.split(stopping_words)[0]
        logger.info(f"-----------------\n{out}", verbose=self.verbose)
        return out
    
    def set_api(self, api: Any, **kwargs) -> None:
        if 'max_new_tokens' not in kwargs:
            kwargs['max_new_tokens']=100
        self.kwargs = kwargs
        self.api = api
        logger.info("done")

    def set_kwargs(self, **kwargs) -> None:
        logger.info(f"set kwargs:{kwargs}")
        if 'max_new_tokens' not in kwargs:
            kwargs['max_new_tokens']=100
        self.kwargs = kwargs

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
    

class VllmLLM(LLM):
    """
    Langchain LLM wrapper for a local LLM
    """
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
        for k in self.kwargs:
            if k not in kwargs:
                kwargs[k] = self.kwargs[k]
        logger.info(f"kwargs:{kwargs}\nstop:{stop}\nprompt:{prompt}", verbose=self.verbose)
        generate_args = {k:kwargs[k] for k in kwargs if k in ['n','max_tokens','temperature','stream']}
        out = self.api.generate(prompt, **generate_args)
        if len(stop)!=0:
            for stopping_words in stop:
                if stopping_words in out:
                    out = out.split(stopping_words)[0]
        logger.info(f"-----------------\n{out}", verbose=self.verbose)
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
    llm = ColossalLLM(n=10)
    model_path = os.environ.get('ZH_MODEL_PATH')
    llm.set_llm("chatglm2", model_path)
    logger.info(llm("Question: what dog doesn't bark? Answer:", max_new_tokens=10))