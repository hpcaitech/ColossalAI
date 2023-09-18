'''
Code for neural text splitter

This code incorporates an algorithm based on:
  Author: miraged3
  Title: "chatchat-space/Langchain-Chatchat"
  URL: https://github.com/chatchat-space/Langchain-Chatchat
'''
import re
from typing import List
from modelscope.pipelines import pipeline
from langchain.text_splitter import CharacterTextSplitter

class NeuralTextSplitter(CharacterTextSplitter):
    '''
    Neural text splitter powered by Ali's document segmentation model
    https://modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base/quickstart
    '''
    def __init__(self, pdf: bool = False, device: str='cpu', **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.device = device
        self.pipeline = pipeline(task="document-segmentation",
            model='damo/nlp_bert_document-segmentation_chinese-base',
            device=self.device)

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        
        result = self.pipeline(documents=text)
        splited_text = [s for s in result["text"].split("\n\t") if s]
        return splited_text
