import argparse
import os

from utils import jload, jdump


def generate(args):
    dataset = []
    for i in range(args.shards):
        shard = jload(os.path.join(args.answer_path,
                      f'{args.model_name}_answers_rank{i}.json'))
        dataset.extend(shard)

    dataset.sort(key=lambda x: x['id'])
    jdump(dataset, os.path.join(args.answer_path,
                                f'{args.model_name}_answers.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--answer_path', type=str, default="answer")
    args = parser.parse_args()
    generate(args)
