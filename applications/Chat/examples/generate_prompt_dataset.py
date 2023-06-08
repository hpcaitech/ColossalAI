import argparse

import random
import json

random.seed(42)


def sample(args):
    with open(args.dataset_path, mode='r') as f:
        dataset_list = json.load(f)

    sampled_dataset = [{"instruction": sample["instruction"], "id":idx}
                       for idx, sample in enumerate(random.sample(dataset_list, args.sample_size))]

    with open(args.save_path, mode='w') as f:
        json.dump(sampled_dataset, f, indent=4,
                  default=str, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None,
                        required=True, help="path to the pretrain dataset")
    parser.add_argument('--save_path', type=str, default='prompt.json',
                        help="path to save the prompt dataset")
    parser.add_argument('--sample_size', type=int,
                        default=16384, help="size of the prompt dataset")
    args = parser.parse_args()
    sample(args)
