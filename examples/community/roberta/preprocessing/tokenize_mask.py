import argparse
import multiprocessing
import os
import time
from random import shuffle

import h5py
import numpy as np
import psutil
from get_mask import PreTrainingDataset
from tqdm import tqdm
from transformers import AutoTokenizer


def get_raw_instance(document, max_sequence_length=512):
    """
    Get the initial training instances, split the whole segment into multiple parts according to the max_sequence_length, and return as multiple processed instances.
    :param document: document
    :param max_sequence_length:
    :return: a list. each element is a sequence of text
    """
    # document = self.documents[index]
    max_sequence_length_allowed = max_sequence_length - 2
    # document = [seq for seq in document if len(seq)<max_sequence_length_allowed]
    sizes = [len(seq) for seq in document]

    result_list = []
    curr_seq = []
    sz_idx = 0
    while sz_idx < len(sizes):
        if len(curr_seq) + sizes[sz_idx] <= max_sequence_length_allowed:  # or len(curr_seq)==0:
            curr_seq += document[sz_idx]
            sz_idx += 1
        elif sizes[sz_idx] >= max_sequence_length_allowed:
            if len(curr_seq) > 0:
                result_list.append(curr_seq)
            curr_seq = []
            result_list.append(document[sz_idx][:max_sequence_length_allowed])
            sz_idx += 1
        else:
            result_list.append(curr_seq)
            curr_seq = []

    if len(curr_seq) > max_sequence_length_allowed / 2:  # /2
        result_list.append(curr_seq)

    # num_instance=int(len(big_list)/max_sequence_length_allowed)+1
    # print("num_instance:",num_instance)

    # result_list=[]
    # for j in range(num_instance):
    #     index=j*max_sequence_length_allowed
    #     end_index=index+max_sequence_length_allowed if j!=num_instance-1 else -1
    #     result_list.append(big_list[index:end_index])
    return result_list


def split_numpy_chunk(path, tokenizer, pretrain_data, host):
    documents = []
    instances = []

    s = time.time()
    with open(path, encoding="utf-8") as fd:
        document = []
        for i, line in enumerate(tqdm(fd)):
            line = line.strip()
            # document = line
            # if len(document.split("<sep>")) <= 3:
            #     continue
            if len(line) > 0 and line[:2] == "]]":  # This is end of document
                documents.append(document)
                document = []
            elif len(line) >= 2:
                document.append(line)
        if len(document) > 0:
            documents.append(document)
    print("read_file ", time.time() - s)

    # documents = [x for x in documents if x]
    # print(len(documents))
    # print(len(documents[0]))
    # print(documents[0][0:10])

    ans = []
    for docs in tqdm(documents):
        ans.append(pretrain_data.tokenize(docs))
    print(time.time() - s)
    del documents

    instances = []
    for a in tqdm(ans):
        raw_ins = get_raw_instance(a)
        instances.extend(raw_ins)
    del ans

    print("len instance", len(instances))

    sen_num = len(instances)
    seq_len = 512
    input_ids = np.zeros([sen_num, seq_len], dtype=np.int32)
    input_mask = np.zeros([sen_num, seq_len], dtype=np.int32)
    segment_ids = np.zeros([sen_num, seq_len], dtype=np.int32)
    masked_lm_output = np.zeros([sen_num, seq_len], dtype=np.int32)

    for index, ins in tqdm(enumerate(instances)):
        mask_dict = pretrain_data.create_training_instance(ins)
        input_ids[index] = mask_dict[0]
        input_mask[index] = mask_dict[1]
        segment_ids[index] = mask_dict[2]
        masked_lm_output[index] = mask_dict[3]

    with h5py.File(f"/output/{host}.h5", "w") as hf:
        hf.create_dataset("input_ids", data=input_ids)
        hf.create_dataset("input_mask", data=input_ids)
        hf.create_dataset("segment_ids", data=segment_ids)
        hf.create_dataset("masked_lm_positions", data=masked_lm_output)

    del instances


def split_numpy_chunk_pool(input_path, output_path, pretrain_data, worker, dupe_factor, seq_len, file_name):
    if os.path.exists(os.path.join(output_path, f"{file_name}.h5")):
        print(f"{file_name}.h5 exists")
        return

    documents = []
    instances = []

    s = time.time()
    with open(input_path, "r", encoding="utf-8") as fd:
        document = []
        for i, line in enumerate(tqdm(fd)):
            line = line.strip()
            if len(line) > 0 and line[:2] == "]]":  # This is end of document
                documents.append(document)
                document = []
            elif len(line) >= 2:
                document.append(line)
        if len(document) > 0:
            documents.append(document)
    print(f"read_file cost {time.time() - s}, length is {len(documents)}")

    ans = []
    s = time.time()
    pool = multiprocessing.Pool(worker)
    encoded_doc = pool.imap_unordered(pretrain_data.tokenize, documents, 100)
    for index, res in tqdm(enumerate(encoded_doc, start=1), total=len(documents), colour="cyan"):
        ans.append(res)
    pool.close()
    print((time.time() - s) / 60)
    del documents

    instances = []
    for a in tqdm(ans, colour="MAGENTA"):
        raw_ins = get_raw_instance(a, max_sequence_length=seq_len)
        instances.extend(raw_ins)
    del ans

    print("len instance", len(instances))

    new_instances = []
    for _ in range(dupe_factor):
        for ins in instances:
            new_instances.append(ins)

    shuffle(new_instances)
    instances = new_instances
    print("after dupe_factor, len instance", len(instances))

    sentence_num = len(instances)
    input_ids = np.zeros([sentence_num, seq_len], dtype=np.int32)
    input_mask = np.zeros([sentence_num, seq_len], dtype=np.int32)
    segment_ids = np.zeros([sentence_num, seq_len], dtype=np.int32)
    masked_lm_output = np.zeros([sentence_num, seq_len], dtype=np.int32)

    s = time.time()
    pool = multiprocessing.Pool(worker)
    encoded_docs = pool.imap_unordered(pretrain_data.create_training_instance, instances, 32)
    for index, mask_dict in tqdm(enumerate(encoded_docs), total=len(instances), colour="blue"):
        input_ids[index] = mask_dict[0]
        input_mask[index] = mask_dict[1]
        segment_ids[index] = mask_dict[2]
        masked_lm_output[index] = mask_dict[3]
    pool.close()
    print((time.time() - s) / 60)

    with h5py.File(os.path.join(output_path, f"{file_name}.h5"), "w") as hf:
        hf.create_dataset("input_ids", data=input_ids)
        hf.create_dataset("input_mask", data=input_mask)
        hf.create_dataset("segment_ids", data=segment_ids)
        hf.create_dataset("masked_lm_positions", data=masked_lm_output)

    del instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True, default=10, help="path of tokenizer")
    parser.add_argument("--seq_len", type=int, default=512, help="sequence length")
    parser.add_argument(
        "--max_predictions_per_seq", type=int, default=80, help="number of shards, e.g., 10, 50, or 100"
    )
    parser.add_argument("--input_path", type=str, required=True, help="input path of shard which has split sentence")
    parser.add_argument("--output_path", type=str, required=True, help="output path of h5 contains token id")
    parser.add_argument(
        "--backend", type=str, default="python", help="backend of mask token, python, c++, numpy respectively"
    )
    parser.add_argument(
        "--dupe_factor",
        type=int,
        default=1,
        help="specifies how many times the preprocessor repeats to create the input from the same article/document",
    )
    parser.add_argument("--worker", type=int, default=32, help="number of process")
    parser.add_argument("--server_num", type=int, default=10, help="number of servers")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    pretrain_data = PreTrainingDataset(
        tokenizer, args.seq_len, args.backend, max_predictions_per_seq=args.max_predictions_per_seq
    )

    data_len = len(os.listdir(args.input_path))

    for i in range(data_len):
        input_path = os.path.join(args.input_path, f"{i}.txt")
        if os.path.exists(input_path):
            start = time.time()
            print(f"process {input_path}")
            split_numpy_chunk_pool(
                input_path, args.output_path, pretrain_data, args.worker, args.dupe_factor, args.seq_len, i
            )
            end_ = time.time()
            print("memory：%.4f GB" % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            print(f"has cost {(end_ - start) / 60}")
            print("-" * 100)
            print("")

    # if you have multiple server, you can use code below or modify code to openmpi

    # host = int(socket.gethostname().split('GPU')[-1])
    # for i in range(data_len // args.server_num + 1):
    #     h = args.server_num * i + host - 1
    #     input_path = os.path.join(args.input_path, f'{h}.txt')
    #     if os.path.exists(input_path):
    #         start = time.time()
    #         print(f'I am server {host}, process {input_path}')
    #         split_numpy_chunk_pool(input_path,
    #                                 args.output_path,
    #                                 pretrain_data,
    #                                 args.worker,
    #                                 args.dupe_factor,
    #                                 args.seq_len,
    #                                 h)
    #         end_ = time.time()
    #         print(u'memory：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    #         print(f'has cost {(end_ - start) / 60}')
    #         print('-' * 100)
    #         print('')
