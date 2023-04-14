import copy
import json
import time
from typing import Dict, List, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import gc
IGNORE_INDEX = -100


def _tokenize_fn(strings: Sequence[str], tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer, max_length) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class EasySupervisedDataset(Dataset):

    def __init__(self, data_file: str, tokenizer: AutoTokenizer, max_length: int = 512) -> None:
        super(EasySupervisedDataset, self).__init__()
        with open(data_file, "r", encoding="UTF-8") as f:
            all_lines = f.readlines()
        #split to source and target ,source the characters before "回答：" including "回答：", target the characters after "回答："
        sources, targets = [], []
        random_print_legal=3
        random_print_illegal=3
        import random
        for line in all_lines:
            if "\t回答\t" in line:
                sep_index = line.index("\t回答\t")
                sources.append(line[:sep_index + len("\t回答\t")])
                targets.append(line[sep_index + len("\t回答\t"):] + tokenizer.eos_token)
                if random_print_legal>0:
                    if random.random()>0.8:
                        print("legal line :",line)
                        random_print_legal-=1
            else:
                sources.append(line)
                targets.append("" + tokenizer.eos_token)
                if random_print_illegal>0:
                    if random.random()>0.8:
                        print("illegal line :",line)
                        random_print_illegal-=1
        data_dict = preprocess(sources, targets, tokenizer, max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.data_file = data_file

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __repr__(self):
        return f"LawSupervisedDataset(data_file={self.data_file}, input_ids_len={len(self.input_ids)}, labels_len={len(self.labels)})"

    def __str__(self):
        return f"LawSupervisedDataset(data_file={self.data_file}, input_ids_len={len(self.input_ids)}, labels_len={len(self.labels)})"


class EasyPromptsDataset(Dataset):

    def __init__(self, data_file: str, tokenizer: AutoTokenizer, max_length: int = 96) -> None:
        super(EasyPromptsDataset, self).__init__()
        with open(data_file, "r", encoding="UTF-8") as f:
            all_lines = f.readlines()
            all_lines = [line if "\t回答\t" not in line else line[:line.index("\t回答\t") + len("\t回答\t")] for line in all_lines]
        self.prompts = [
            tokenizer(line, return_tensors='pt', max_length=max_length, padding='max_length',
                      truncation=True)['input_ids'].to(torch.cuda.current_device()).squeeze(0)
            for line in tqdm(all_lines)
        ]
        self.data_file = data_file

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

    def __repr__(self):
        return f"LawPromptsDataset(data_file={self.data_file}, prompts_len={len(self.prompts)})"

    def __str__(self):
        return f"LawPromptsDataset(data_file={self.data_file}, prompts_len={len(self.prompts)})"

from multiprocessing import Pool, Process,Queue
from contextlib import closing
import os
class EasyRewardDataset(Dataset):

    def __init__(self, train_file: str, tokenizer: AutoTokenizer, special_token=None, max_length=512,concurrency = 20) -> None:
        super(EasyRewardDataset, self).__init__()
        self.current_index = 0   
        self.chosen = []
        self.reject = []
        if special_token is None:
            self.end_token = tokenizer.eos_token
        else:
            self.end_token = special_token
        #read all lines in the train_file to a list
        with open(train_file, "r", encoding="UTF-8") as f:
            all_lines = f.readlines()
        self.concurrency = concurrency
        max_tasks_in_queue = 100
        from pyarrow import parquet as pq
        from pandas import DataFrame
        import pandas as pd
        def mp_function(task_queue:Queue,index:int):
            #get a task from task_queue, catch the exception if the queue is closed
            parquet_dir = 'parquet'
            if not os.path.exists(parquet_dir):
                os.makedirs(parquet_dir,exist_ok=True)
            index = 0
            df = DataFrame(columns=['chosen_input_ids','chosen_attention_mask','reject_input_ids','reject_attention_mask'])
            while True:
                try:
                    line = task_queue.get()
                    if line == 'Exit':
                        break
                    data = json.loads(line)
                    prompt = "请根据法律回答。提问：" + data['prompt'] + "\t回答：\t"
                    chosen = prompt + data['chosen'] + self.end_token
                    chosen_token = tokenizer(chosen,
                                                    max_length=max_length,
                                                    padding="max_length",
                                                    truncation=True,
                                                    return_tensors="pt")
                    reject = prompt + data['rejected'] + self.end_token
                    reject_token = tokenizer(reject,
                                                    max_length=max_length,
                                                    padding="max_length",
                                                    truncation=True,
                                                    return_tensors="pt")
                    data = {
                        "chosen_input_ids": chosen_token['input_ids'].tolist(),
                        "chosen_attention_mask": chosen_token['attention_mask'].tolist(),
                        "reject_input_ids": reject_token['input_ids'].tolist(),
                        "reject_attention_mask": reject_token['attention_mask'].tolist()
                    }
                    new_df = DataFrame.from_dict(data)
                    df = pd.concat([df,new_df],ignore_index=True)
                    if len(df) > 100:
                        print(f"Write {len(df)} to parquet file {parquet_dir}/{os.getpid()}_{index}.parquet}}")
                        df.to_parquet(f'{parquet_dir}/{os.getpid()}_{index}.parquet')
                        index += 1
                        df = DataFrame(columns=['chosen_input_ids','chosen_attention_mask','reject_input_ids','reject_attention_mask'])
                    #print the process id and the task
                    # print(f"Process {os.getpid()} has processed task")
                except:
                    print(f"Process {os.getpid()} has no more tasks")
                    break

            if len(df) > 0:
                print(f"Write {len(df)} to parquet file {parquet_dir}/{os.getpid()}_{index}.parquet}}")
                df.to_parquet(f'{parquet_dir}/{os.getpid()}_{index}.parquet')
        processes = []
        task_queues = []
        for i in range(concurrency):
            task_queue = Queue(max_tasks_in_queue)
            p = Process(target=mp_function, args=(task_queue,i))
            p.start()
            processes.append(p)    
            task_queues.append(task_queue)
        index = 0
        for line in tqdm(all_lines):
            task_queues[index % concurrency].put(line)
            index += 1
            
        print('waiting for all tasks to be done')
        for task_queue in task_queues:
            task_queue.put("Exit")
        for task_queue in task_queues:
            while True:
                if task_queue.empty():
                    print('all tasks are done, closing the queue')
                    task_queue.close()
                    break
                print(f'still have {task_queue.qsize()} tasks to be done,waiting for 1s')
                time.sleep(1)
        print('get all results from processes')
        time.sleep(5)
        print(f"waiting for response in main process {os.getpid()}")
        #terminate all processes
        for p in processes:
            p.terminate()
            p.join()
        print('all tasks done')

    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return self.chosen[idx]["input_ids"], self.chosen[idx]["attention_mask"], self.reject[idx]["input_ids"], self.reject[idx]["attention_mask"]
    


    def gen(self):
        #yeild from the dataset
        while self.current_index < len(self.chosen):
            chosen_ids, c_mask, reject_ids, r_mask =self.chosen[self.current_index]["input_ids"], self.chosen[self.current_index]["attention_mask"], self.reject[self.current_index]["input_ids"], self.reject[self.current_index]["attention_mask"]
            yield dict(chosen_input_ids=chosen_ids, chosen_attention_mask=c_mask, reject_input_ids=reject_ids, reject_attention_mask=r_mask)
            self.current_index += 1

    @staticmethod
    def collate_fn(batch):
        size = len(batch)
        chosen_input_ids = [d['chosen_input_ids'] for d in batch]
        chosen_attention_mask = [d['chosen_attention_mask'] for d in batch]
        reject_input_ids = [d['reject_input_ids'] for d in batch]
        reject_attention_mask = [d['reject_attention_mask'] for d in batch]

        return (torch.tensor(chosen_input_ids,dtype=torch.long), 
                torch.tensor(chosen_attention_mask), torch.tensor(reject_input_ids,dtype=torch.long), torch.tensor(reject_attention_mask))

    #python representation of the object and the string representation of the object
    def __repr__(self):
        return f"LawRewardDataset(chosen_len={len(self.chosen)}, reject_len={len(self.reject)})"

    def __str__(self):
        return f"LawRewardDataset(chosen_len={len(self.chosen)}, reject_len={len(self.reject)})"


'''
Easy SFT just accept a text file which can be read line by line. However the datasest will group texts together to max_length so LLM will learn the texts meaning better.
If individual lines are not related, just set is_group_texts to False.
'''
def split_texts(input_text :str, max_length) -> List[str]:
    chinese_punctuations = "，。！？；："
    english_punctuations = ",.!?;:"
    punctuations = chinese_punctuations + english_punctuations
    texts = []
    #first split the text by punctuations
    for punctuation in punctuations:
        input_text = input_text.replace(punctuation, "\t" + punctuation + "\t")
    #then split the text by \t
    input_text = input_text.split("\t")
    #remove empty strings
    input_text = [text for text in input_text if text.strip() != ""]
    #group texts together
    current_text = ""
    for text in input_text:
        if len(current_text) + len(text) > max_length:
            #if current_text is still too long, just split it by max_length
            if len(current_text) > max_length:
                for i in range(0, len(current_text), max_length):
                    texts.append(current_text[i:i + max_length])
            else:
                texts.append(current_text)
            current_text = text
        else:
            current_text += text
    if current_text != "":
        texts.append(current_text)
    return texts 
class EasySFTDataset(Dataset):

    def __init__(self, data_file: str, tokenizer: AutoTokenizer, max_length=512, is_group_texts=True,concurrency = 20,mask_prompts =  False,prompts_sep :str = "\t回答\t") -> None:
        super().__init__()
        #read the data_file line by line
        with open(data_file, "r", encoding="UTF-8") as f:
            all_lines = f.readlines()
        print(f"total lines: {len(all_lines)}")
        if is_group_texts:
            grouped_lines = []
            current_line = ""
            for line in all_lines:
                line = line.strip()
                current_line += line
                if len(current_line) > max_length:
                    splitted_lines = split_texts(current_line, max_length)
                    grouped_lines.extend(splitted_lines)
                    current_line = ""
        else:
            grouped_lines = []
            for line in all_lines:
                line = line.strip()
                if not mask_prompts:
                    splitted_lines = split_texts(line, max_length)
                    grouped_lines.extend(splitted_lines)
                else:
                    #if the line does not contain the prompts_sep, just skip it
                    if prompts_sep not in line:
                        continue
                    #if the length before prompts_sep is too long, just skip it
                    if len(line.split(prompts_sep)[0]) > max_length-len(prompts_sep):
                        print(f'prompts is too long, skip this line: {line}')
                        continue
                    grouped_lines.append(line)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print(f"total lines: {len(grouped_lines)}")
        print('samples from grouped_lines')
        print(grouped_lines[:10])
        
        from pyarrow import parquet as pq
        from pandas import DataFrame
        import pandas as pd
        def mp_function(task_queue:Queue,index:int):
            #get a task from task_queue, catch the exception if the queue is closed
            parquet_dir = 'parquet'
            if not os.path.exists(parquet_dir):
                os.makedirs(parquet_dir,exist_ok=True)
            index = 0
            df = DataFrame(columns=['input_ids','labels','attention_mask'])
            if mask_prompts:
                prompts_sep_ids = tokenizer.encode(prompts_sep,add_special_tokens=False)[1:]
            while True:
                try:
                    line = task_queue.get()
                    if line == 'Exit':
                        break
                    token = tokenizer(line,
                                                    max_length=max_length,
                                                    padding="max_length",
                                                    truncation=True,
                                                    return_tensors="pt")
                    labels = token['input_ids'].tolist()[0]
                    labels = [-100 if label == tokenizer.pad_token_id else label for label in labels]
                    if mask_prompts:
                        def find_first_sub_list(full_list, sub_list):
                            index = -1
                            len_of_full = len(full_list)
                            len_of_sub = len(sub_list)
                            for i in range(len_of_full):
                                if full_list[i:i+len_of_sub] == sub_list:
                                    index = i
                                    break
                            return index
                        prompts_sep_index = find_first_sub_list(token['input_ids'].tolist()[0],prompts_sep_ids)
                        if prompts_sep_index != -1:
                            #mask the ids before prompts_sep
                            labels = [-100 if i < prompts_sep_index else label for i,label in enumerate(labels)]
                        else:
                            print(f'wrong prompts_sep {prompts_sep} in {line}, {labels}')

                    labels = [labels]
                    data = {
                        "input_ids": token['input_ids'].tolist(),
                        "labels": labels,
                        "attention_mask": token['attention_mask'].tolist()
                    }
                    new_df = DataFrame.from_dict(data)
                    df = pd.concat([df,new_df],ignore_index=True)
                    if len(df) > 100:
                        print(f"Write {len(df)} to parquet file {parquet_dir}/{os.getpid()}_{index}.parquet}}")
                        df.to_parquet(f'{parquet_dir}/{os.getpid()}_{index}.parquet')
                        index += 1
                        df = DataFrame(columns=['input_ids','labels','attention_mask'])
                    #print the process id and the task
                    # print(f"Process {os.getpid()} has processed task")
                except:
                    print(f"Process {os.getpid()} has no more tasks")
                    break

            if len(df) > 0:
                print(f"Write {len(df)} to parquet file {parquet_dir}/{os.getpid()}_{index}.parquet}}")
                df.to_parquet(f'{parquet_dir}/{os.getpid()}_{index}.parquet')
        from tqdm import tqdm
        progress_bar = tqdm(grouped_lines, desc="Tokenizing texts")
        processes = []
        task_queues = []
        self.concurrency = concurrency
        max_tasks_in_queue = 100
        for i in range(concurrency):
            task_queue = Queue(max_tasks_in_queue)
            p = Process(target=mp_function, args=(task_queue,i))
            p.start()
            processes.append(p)    
            task_queues.append(task_queue)
        index = 0
        for line in progress_bar:
            task_queues[index % concurrency].put(line)
            index += 1
        print('waiting for all tasks to be done')
        for task_queue in task_queues:
            task_queue.put("Exit")
        for task_queue in task_queues:
            while True:
                if task_queue.empty():
                    print('all tasks are done, closing the queue')
                    task_queue.close()
                    break
                print(f'still have {task_queue.qsize()} tasks to be done,waiting for 1s')
                time.sleep(1)
        print('get all results from processes')
        time.sleep(5)
        print(f"waiting for response in main process {os.getpid()}")
        #terminate all processes
        for p in processes:
            p.terminate()
            p.join()
        print('all tasks done')
        self.input_ids = []
        self.labels = []
        self.file_name = data_file
        self.attention_mask = []
    
    @staticmethod
    def collate_fn(batch):
        size = len(batch)
        input_ids = [d['input_ids'] for d in batch]
        labels = [d['labels'] for d in batch]
        attention_mask = [d['attention_mask'] for d in batch]

        return dict(input_ids=torch.tensor(input_ids,dtype=torch.long), 
                attention_mask=torch.tensor(attention_mask), labels=torch.tensor(labels,dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    #get item from dataset
    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx], attention_mask=self.attention_mask[idx])

    #generate the dataset description to be printed by print in python
    def __repr__(self):
        return f"EasySFTDataset(len={len(self)},\nfile_name is {self.file_name})"

    #generate the dataset description to be printed by print in python
    def __str__(self):
        return f"EasySFTDataset(len={len(self)},\nfile_name is {self.file_name})"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--is_group_texts", action='store_true', default=False)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--need_trust_code", action='store_true',default=False)
    parser.add_argument("--dataset_type", type=str, default="easy_sft")
    parser.add_argument("--task_type", type=str, default="convert_2_binary")
    parser.add_argument("--mask_prompts", action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    if args.task_type == "convert_2_binary":
        from transformers import AutoTokenizer
        if args.need_trust_code:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

        if args.dataset_type == "easy_sft":
            dataset = EasySFTDataset(args.input_file, tokenizer, max_length =args.max_length, is_group_texts= args.is_group_texts,mask_prompts=args.mask_prompts)
        elif args.dataset_type == "reward":
            dataset = EasyRewardDataset(args.input_file, tokenizer, max_length = args.max_length)
        
        #save the dataset
        print(f'Finish loading and now saving...to {args.output_file}')
        import datasets
        dir = 'parquet'
        datafiles = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.parquet')]
        ds = datasets.load_dataset('parquet', data_files=datafiles)
        ds.save_to_disk(args.output_file)
        #delete all the temporary files
        for file in datafiles:
            os.remove(file)
    elif args.task_type == "load_from_binary":
        import datasets
        ds_train = datasets.load_from_disk(args.output_file)
        print(ds_train)

        from torch.utils.data import DataLoader
        if args.dataset_type == "reward":
            dataloader = DataLoader(ds_train, batch_size=1, shuffle=True,collate_fn=EasyRewardDataset.collate_fn)
            print(dataloader)
            for chosen_ids, c_mask, reject_ids, r_mask in dataloader:
                print(chosen_ids.tolist())
                print(chosen_ids.shape)
                print(c_mask.shape)
                print(reject_ids.shape)
                print(r_mask.shape)
                break
        elif args.dataset_type == "easy_sft":
            dataloader = DataLoader(ds_train, batch_size=1, shuffle=True,collate_fn=EasySFTDataset.collate_fn)
            print(dataloader)
            for batch in dataloader:
                input_ids = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch['attention_mask']
                print(input_ids.tolist())
                print(input_ids.shape)
                print(labels.shape)
                print(attention_mask.shape)
                print(labels.tolist())
                break