import argparse
import functools
import json
import multiprocessing
import os
import re
import time
from typing import List

from tqdm import tqdm


def split_sentence(document: str, flag: str = "all", limit: int = 510) -> List[str]:
    sent_list = []
    try:
        if flag == "zh":
            document = re.sub("(?P<quotation_mark>([。？！…](?![”’\"'])))", r"\g<quotation_mark>\n", document)
            document = re.sub("(?P<quotation_mark>([。？！]|…{1,2})[”’\"'])", r"\g<quotation_mark>\n", document)
        elif flag == "en":
            document = re.sub("(?P<quotation_mark>([.?!](?![”’\"'])))", r"\g<quotation_mark>\n", document)
            document = re.sub(
                "(?P<quotation_mark>([?!.][\"']))", r"\g<quotation_mark>\n", document
            )  # Special quotation marks
        else:
            document = re.sub("(?P<quotation_mark>([。？！….?!](?![”’\"'])))", r"\g<quotation_mark>\n", document)

            document = re.sub(
                "(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’\"']))", r"\g<quotation_mark>\n", document
            )  # Special quotation marks

        sent_list_ori = document.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            elif len(sent) <= 2:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:
        sent_list.clear()
        sent_list.append(document)
    return sent_list


def get_sent(output_path, input_path, fin_list=[], host=-1, seq_len=512) -> None:
    workers = 32

    if input_path[-1] == "/":
        input_path = input_path[:-1]

    cur_path = os.path.join(output_path, str(host) + ".txt")
    new_split_sentence = functools.partial(split_sentence, limit=seq_len - 2)
    with open(cur_path, "w", encoding="utf-8") as f:
        for fi, fin_path in enumerate(fin_list):
            if not os.path.exists(os.path.join(input_path, fin_path[0])):
                continue
            if ".json" not in fin_path[0]:
                continue

            print("Processing ", fin_path[0], " ", fi)

            with open(os.path.join(input_path, fin_path[0]), "r") as fin:
                f_data = [l["content"] for l in json.load(fin)]

                pool = multiprocessing.Pool(workers)
                all_sent = pool.imap_unordered(new_split_sentence, f_data, 32)
                pool.close()
            print("finished..")

            cnt = 0
            for d in tqdm(all_sent):
                for i in d:
                    f.write(i.strip() + "\n")
                f.write("]]" + "\n")
                cnt += 1
                # if cnt >= 2:
                #     exit()


def getFileSize(filepath, shard):
    all_data = []
    for i in os.listdir(filepath):
        all_data.append(os.path.join(filepath, i))
    all_size = sum([os.path.getsize(os.path.join(filepath, f)) for f in all_data])
    ans = [[f.split("/")[-1], os.path.getsize(os.path.join(filepath, f))] for f in all_data]
    ans = sorted(ans, key=lambda x: x[1], reverse=True)
    per_size = all_size / shard
    real_shard = []
    temp = []
    accu_size = 0
    for i in ans:
        accu_size += i[1]
        temp.append(i)
        if accu_size > per_size:
            real_shard.append(temp)
            accu_size = 0
            temp = []

    if len(temp) > 0:
        real_shard.append(temp)

    return real_shard


def get_start_end(real_shard, base=0, server_num=10, server_name="GPU"):
    import socket

    host = int(socket.gethostname().split(server_name)[-1])

    fin_list = real_shard[server_num * base + host - 1]
    print(fin_list)
    print(f"I am server {host}, process {server_num * base + host - 1}, len {len(fin_list)}")
    return fin_list, host


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_num", type=int, default=10, help="number of servers")
    parser.add_argument("--seq_len", type=int, default=512, help="sequence length")
    parser.add_argument("--shard", type=int, default=100, help="number of shards, e.g., 10, 50, or 100")
    parser.add_argument("--input_path", type=str, required=True, help="input path of original corpus")
    parser.add_argument("--output_path", type=str, required=True, help="output path of shard which has split sentence")
    args = parser.parse_args()

    server_num = args.server_num
    seq_len = args.seq_len
    shard = args.shard
    input_path = args.input_path
    output_path = args.output_path

    real_shard = getFileSize(input_path, shard)

    start = time.time()
    for index, shard in enumerate(real_shard):
        get_sent(output_path, input_path, fin_list=shard, host=index, seq_len=seq_len)
    print(f"cost {str(time.time() - start)}")

    # if you have multiple server, you can use code below or modify code to openmpi

    # for i in range(len(real_shard) // server_num + 1):
    #     fin_list, host = get_start_end(real_shard, i)

    #     start = time.time()
    #     get_sent(output_path,
    #             input_path,
    #             fin_list=fin_list, host= 10 * i + host - 1)

    #     print(f'cost {str(time.time() - start)}')
