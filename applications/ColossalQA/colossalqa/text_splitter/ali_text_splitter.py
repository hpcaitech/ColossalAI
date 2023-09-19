'''
Code for neural text splitter
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
        '''
        Initiate neural text spliter. Since the nerual document segmentation model has a max 
        token limitation of 1024 tokens. We need to chuncate the input text first.
        Args:
            pdf: whether the input text is pdf format
            device: cpu or cuda device
            separator: a string, used to force separate two chunks, default is '\n\n'
        '''
        super().__init__(**kwargs)
        self.pdf = pdf
        self.device = device
        self.pipeline = pipeline(task="document-segmentation",
            model='damo/nlp_bert_document-segmentation_chinese-base',
            device=self.device)

    def split_text(self, text: str) -> List[str]:
        """Return the list of separated text chunks"""
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        splited_text = []
        chunks = super().split_text(text)
        if len(chunks)>1:
            for chunk in chunks:
                splited_text.extend(self.split_text(chunk))
        else:
            result = self.pipeline(documents=chunks[0])
            splited_text = [s.strip() for s in result["text"].split("\n\t") if s]
        return splited_text
