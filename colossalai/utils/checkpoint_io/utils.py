# xxx.pt -> global checkpoint
# xxx.chunk0.pt -> global checkpoint chunk
# xxx.chunk0.0.pt -> checkpoint chunk for process 0

import os
import re
from typing import List
import argparse

DIST_CHUNK_PAT = re.compile(r'(.+)\.chunk\d+\.\d+\.pt')
GLOBAL_CHUNK_PAT = re.compile(r'(.+)\.chunk\d+\.pt')
DIST_CHUNK_TEMPLATE = '{}\.chunk\d+\.\d+\.pt'
GLOBAL_CHUNK_TEMPLATE = '{}\.chunk\d+\.pt'


def extract_checkpoint_template(path: str) -> str:
    filename = os.path.split(path)[1]
    res = DIST_CHUNK_PAT.match(filename)
    if res:
        return DIST_CHUNK_TEMPLATE.format(res[1])
    res = GLOBAL_CHUNK_PAT.match(filename)
    if res:
        return GLOBAL_CHUNK_TEMPLATE.format(res[1])
    return filename


def find_checkpoints(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise OSError(f'{path} is not a file')
    checkpoint_template = extract_checkpoint_template(path)
    file_dir = os.path.dirname(path)
    targets = []
    for name in os.listdir(file_dir):
        target_path = os.path.join(file_dir, name)
        if not os.path.isfile(target_path):
            continue
        if re.match(checkpoint_template, name):
            targets.append(target_path)
    return targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    print(find_checkpoints(args.path))
