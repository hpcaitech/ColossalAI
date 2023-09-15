# -----------------------------------------------------------------------------
# This code incorporates an algorithm based on:
#   Author: miraged3
#   Title: "chatchat-space/Langchain-Chatchat"
#   URL: https://github.com/chatchat-space/Langchain-Chatchat
# -----------------------------------------------------------------------------

from modelscope.pipelines import pipeline
from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List

class NeuralTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, device: str='cpu', **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.device = device

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        p = pipeline(task="document-segmentation",
            model='damo/nlp_bert_document-segmentation_chinese-base',
            device=self.device)
        result = p(documents=text)
        splited_text = [s for s in result["text"].split("\n\t") if s]
        return splited_text
