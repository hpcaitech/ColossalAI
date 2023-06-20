# MIT License

# Copyright (c) 2022 Ming Zhong

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


class UniEvaluator:

    def __init__(self, model_name_or_path, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up model """
        self.device = device
        self.max_length = max_length

        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config, cache_dir=cache_dir)

        self.model.eval()
        self.model.to(device)

        self.softmax = nn.Softmax(dim=1)

        self.pos_id = self.tokenizer("Yes")["input_ids"][0]
        self.neg_id = self.tokenizer("No")["input_ids"][0]

    def score(self, inputs, task, category, dim, batch_size=8):
        """
            Get scores for the given samples.
            final_score = postive_score / (postive_score + negative_score)
        """

        # The implementation of "forward" in T5 still requires decoder_input_ids.
        # Therefore, we construct a random one-word target sequence.
        # The content of the target has no effect on the final scores.
        tgts = ["No" for _ in range(len(inputs))]

        pos_score_list, neg_score_list = [], []
        for i in tqdm(range(0, len(inputs), batch_size), desc=f"{category}-({dim}-{task}): "):
            src_list = inputs[i:i + batch_size]
            tgt_list = tgts[i:i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(src_list,
                                                 max_length=self.max_length,
                                                 truncation=True,
                                                 padding=True,
                                                 return_tensors='pt')
                    encoded_tgt = self.tokenizer(tgt_list,
                                                 max_length=self.max_length,
                                                 truncation=True,
                                                 padding=True,
                                                 return_tensors='pt')

                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)[:, 0].unsqueeze(-1)

                    output = self.model(input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens)
                    logits = output.logits.view(-1, self.model.config.vocab_size)

                    pos_score = self.softmax(logits)[:, self.pos_id]    # Yes
                    neg_score = self.softmax(logits)[:, self.neg_id]    # No

                    cur_pos_score = [x.item() for x in pos_score]
                    cur_neg_score = [x.item() for x in neg_score]
                    pos_score_list += cur_pos_score
                    neg_score_list += cur_neg_score

            except RuntimeError:
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)

        score_list = []
        for i in range(len(pos_score_list)):
            score_list.append(pos_score_list[i] / (pos_score_list[i] + neg_score_list[i]))

        return score_list
