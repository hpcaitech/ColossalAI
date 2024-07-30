"""
Code for Chinese text splitter
"""

from typing import Any, List, Optional

from colossalqa.text_splitter.utils import get_cleaned_paragraph
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChineseTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, separators: Optional[List[str]] = None, is_separator_regrx: bool = False, **kwargs: Any):
        self._separators = separators or ["\n\n", "\n", "，", "。", "！", "？", "?"]
        if "chunk_size" not in kwargs:
            kwargs["chunk_size"] = 50
        if "chunk_overlap" not in kwargs:
            kwargs["chunk_overlap"] = 10
        super().__init__(separators=separators, keep_separator=True, **kwargs)
        self._is_separator_regex = is_separator_regrx

    def split_text(self, text: str) -> List[str]:
        """Return the list of separated text chunks"""
        cleaned_paragraph = get_cleaned_paragraph(text)
        splitted = []
        for paragraph in cleaned_paragraph:
            segs = super().split_text(paragraph)
            for i in range(len(segs) - 1):
                if segs[i][-1] not in self._separators:
                    pos = text.find(segs[i])
                    pos_end = pos + len(segs[i])
                    if i > 0:
                        last_sentence_start = max([text.rfind(m, 0, pos) for m in ["。", "！", "？"]])
                        pos = last_sentence_start + 1
                        segs[i] = str(text[pos:pos_end])
                    if i != len(segs) - 1:
                        next_sentence_end = max([text.find(m, pos_end) for m in ["。", "！", "？"]])
                        segs[i] = str(text[pos : next_sentence_end + 1])
                splitted.append(segs[i])
        if len(splitted) <= 1:
            return splitted
        splitted_text = []
        i = 1
        if splitted[0] not in splitted[1]:
            splitted_text.append([splitted[0], 0])
        if splitted[-1] not in splitted[-2]:
            splitted_text.append([splitted[-1], len(splitted) - 1])
        while i < len(splitted) - 1:
            if splitted[i] not in splitted[i + 1] and splitted[i] not in splitted[i - 1]:
                splitted_text.append([splitted[i], i])
            i += 1
        splitted_text = sorted(splitted_text, key=lambda x: x[1])
        splitted_text = [splitted_text[i][0] for i in range(len(splitted_text))]
        ret = []
        for s in splitted_text:
            if s not in ret:
                ret.append(s)
        return ret
