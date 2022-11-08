import collections
import glob
import logging
import multiprocessing
import os
import sys

import jieba
import six
import torch
from tools.tokenization_enc_dec import EncDecTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

from colossalai.registry import DATASETS

try:
    import nltk

    nltk_available = True
except ImportError:
    nltk_available = False

jieba.setLogLevel(logging.INFO)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Should be running on Python 3")


class WordpieceTokenizer(object):

    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):

        token = convert_to_unicode(token)

        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]

        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if is_contain_chinese(substr):
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                else:
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
                continue
            sub_tokens.append(cur_substr)
            start = end

        return sub_tokens


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='utf-8') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


class EncDecTokenizer(object):

    def __init__(self, vocab_file, max_len=None, max_sentinels=190):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = load_vocab(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder)

        self.translator = str.maketrans(" \n", "\u2582\u2583")

        self.sentinel_list = [self.encoder['<s_{}>'.format(i)] for i in range(max_sentinels)]

        self.en_vocab = {}
        for k, v in self.encoder.items():
            if is_contain_chinese(k):
                self.en_vocab[v] = False
            else:
                self.en_vocab[v] = True
        self.en_vocab[10] = False

    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return len(self.encoder)

    @property
    def eod_id(self):
        return self.encoder[self.eod_token]

    @property
    def pad_id(self):
        return self.encoder[self.pad_token]

    @property
    def eod_token(self):
        return '<eod>'

    @property
    def pad_token(self):
        return '<pad>'

    def get_sentinel_num(self):
        return len(self.sentinel_list)

    def get_sentinel_id(self, idx):
        return self.sentinel_list[idx]

    def tokenize(self, text):
        """ Tokenize a string. """
        output_tokens = []
        for x in jieba.cut(text, cut_all=False):
            x = x.translate(self.translator)
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))

        # print(output_tokens)

        return output_tokens

    def encode(self, text):
        output_tokens = [self.encoder[x] for x in self.tokenize(text)]

        # filter space
        new_output_tokens = [output_tokens[0]]
        for i, x in enumerate(output_tokens[1:-1]):
            if x == 10:
                if self.en_vocab[output_tokens[i]] and self.en_vocab[output_tokens[i + 2]]:
                    continue
            new_output_tokens.append(x)
        new_output_tokens.append(output_tokens[-1])

        return new_output_tokens

    def decode(self, tokens):
        new_tokens = []
        for i, x in enumerate(tokens[:-1]):
            if self.en_vocab[x] and self.en_vocab[tokens[i + 1]]:
                new_tokens.append(x)
                new_tokens.append(10)
            else:
                new_tokens.append(x)
        new_tokens.append(tokens[-1])

        # text = ''.join([self.decoder[x] for x in new_tokens])
        # text = text.replace('\u2582', ' ').replace('\u2583', '\n')
        # return text
        return [self.decoder[x] for x in tokens]


class IdentitySplitter(object):

    @staticmethod
    def tokenize(*text):
        return text


class Encoder(object):

    def __init__(self, vocab_path, length, sentence_splitter):
        self.vocab_path = vocab_path
        self.length = length
        self.sentence_splitter = sentence_splitter
        self.tokenizer = EncDecTokenizer(os.path.join(self.vocab_path))
        self.splitter = IdentitySplitter()

    def initializer(self):
        # Use Encoder class as a container for global data
        pass

    def encode(self, line):
        # end with <eod>
        if len(line) > 20000:
            return None, 0
        if len(line) < 10:
            return None, 0
        data = line.strip().strip('<n>')
        data = data.replace("<n>", "\n")
        doc_ids = self.tokenizer.encode(data)
        doc_ids.append(self.tokenizer.eod_id)
        return doc_ids, len(line)


@DATASETS.register_module
class YuanDataset(Dataset):
    """
    Yuan is an open source Chinese dataset, which can be accessed on https://github.com/Shawn-Inspur/Yuan-1.0.

    Args:
        path(str): Path to dataset's folder, raw data should be organized under the folder as 001.txt, 002.txt...
                   eg:/path/yuan/dataset
        vocab_path(str): Path to the vocab file. eg:/path/yuan/vocab.txt
        seq_len(int): Sequence length of the transformer, defaults to 2048.
    """

    def __init__(self, path, vocab_path, seq_len=2048) -> None:
        super().__init__()

        self.input_path = path
        workers = 16
        sentence_splitter = None
        self.vocab_path = vocab_path
        self.pad_id = EncDecTokenizer(os.path.join(self.vocab_path)).pad_id
        self.length = seq_len

        if self.input_path[-1] == '/':
            self.input_path = self.input_path[:-1]
        if os.path.exists(os.path.join(self.input_path, 'data_list.pt')):
            self.data_path = torch.load(os.path.join(self.input_path, 'data_list.pt'))
            return

        fin_list = glob.glob(self.input_path + '/0[0-9][0-9].txt')
        self.data_path = []
        for fin_path in fin_list:
            if not os.path.exists(fin_path):
                continue
            if '.txt' not in fin_path:
                continue

            all_data = []
            print("Processing ", fin_path)
            with open(fin_path, 'r', encoding='utf-8', errors='ignore') as fin:

                encoder = Encoder(self.vocab_path, seq_len, sentence_splitter)
                pool = multiprocessing.Pool(workers, initializer=encoder.initializer)
                encoded_docs = pool.imap_unordered(encoder.encode, fin, 30)

                for i, (no_noise_tokens, bytes_processed) in tqdm(enumerate(encoded_docs, start=1)):
                    if no_noise_tokens is None:
                        continue
                    all_data.append(no_noise_tokens)

                pool.close()

            print('Saving ', fin_path)
            base_path = fin_path.replace('.txt', '')
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            idx = 0
            for d in tqdm(all_data):
                idx += 1
                cur_path = os.path.join(base_path, str(idx) + '.txt')
                with open(cur_path, 'w+', encoding='utf-8') as f:
                    for i in d:
                        f.write(str(i) + ' ')
                    f.write('\n')
                self.data_path.append(cur_path.replace(self.input_path + '/', ''))

        torch.save(self.data_path, os.path.join(self.input_path, 'data_list.pt'))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]
        root = os.path.join(self.input_path, path)
        with open(root, "r") as f:
            data = f.readlines()
        assert len(data) == 1
        data = data[0][:-2].split(' ')
        try:
            data = list(map(int, data))
        except:
            while '' in data:
                data.remove('')
            data = list(map(int, data))
        if len(data) > self.length:
            data = data[:self.length - 1] + [data[-1]]
            mask = [1] * self.length
        else:
            data += [self.pad_id] * (self.length - len(data))
            mask = [1] * len(data) + [0] * (self.length - len(data))

        data = torch.tensor(data)
        mask = torch.tensor(mask)
        return {'input_ids': data, 'attention_mask': mask}, data


if __name__ == '__main__':
    dataset = YuanDataset('/data/gpt-yuan/ASC22/dataset', vocab_path='/data/gpt-yuan/ASC22/vocab.txt', seq_len=2048)
    test = dataset.__getitem__(0)
    print(test)
