import argparse
import os
import subprocess
from filecmp import cmp
from functools import cmp_to_key

import requests
from bs4 import BeautifulSoup
from packaging import version

WHEEL_TEXT_ROOT_URL = 'https://github.com/hpcaitech/public_assets/tree/main/colossalai/torch_build/torch_wheels'
RAW_TEXT_FILE_PREFIX = 'https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/torch_build/torch_wheels'
CUDA_HOME = os.environ['CUDA_HOME']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_version', type=str)
    parser.add_argument(
        '--nightly',
        action='store_true',
        help=
        'whether this build is for nightly release, if True, will only build on the latest PyTorch version and Python 3.8'
    )
    return parser.parse_args()


def get_cuda_bare_metal_version():
    raw_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return bare_metal_major, bare_metal_minor


def all_wheel_info():
    page_text = requests.get(WHEEL_TEXT_ROOT_URL).text
    soup = BeautifulSoup(page_text)

    all_a_links = soup.find_all('a')

    wheel_info = dict()

    for a_link in all_a_links:
        if 'cuda' in a_link.text and '.txt' in a_link.text:
            filename = a_link.text
            torch_version, cuda_version = filename.rstrip('.txt').split('-')
            cuda_version = cuda_version.lstrip('cuda')

            if torch_version not in wheel_info:
                wheel_info[torch_version] = dict()
            wheel_info[torch_version][cuda_version] = dict()

            file_text = requests.get(f'{RAW_TEXT_FILE_PREFIX}/{filename}').text
            lines = file_text.strip().split('\n')

            for line in lines:
                parts = line.split('\t')
                method, url, python_version = parts[:3]

                if len(parts) > 3:
                    flags = parts[3]
                    flags = ' '.join(flags.split('+'))
                else:
                    flags = ''
                wheel_info[torch_version][cuda_version][python_version] = dict(method=method, url=url, flags=flags)
    return wheel_info


def build_colossalai(wheel_info):
    cuda_version_major, cuda_version_minor = get_cuda_bare_metal_version()
    cuda_version_on_host = f'{cuda_version_major}.{cuda_version_minor}'

    for torch_version, cuda_versioned_wheel_info in wheel_info.items():
        for cuda_version, python_versioned_wheel_info in cuda_versioned_wheel_info.items():
            if cuda_version_on_host == cuda_version:
                for python_version, wheel_info in python_versioned_wheel_info.items():
                    url = wheel_info['url']
                    method = wheel_info['method']
                    flags = wheel_info['flags']
                    filename = url.split('/')[-1].replace('%2B', '+')
                    cmd = f'bash ./build_colossalai_wheel.sh {method} {url} {filename} {cuda_version} {python_version} {torch_version} {flags}'
                    os.system(cmd)


def main():
    args = parse_args()
    wheel_info = all_wheel_info()

    # filter wheels on condition
    all_torch_versions = list(wheel_info.keys())

    def _compare_version(a, b):
        if version.parse(a) > version.parse(b):
            return 1
        else:
            return -1

    all_torch_versions.sort(key=cmp_to_key(_compare_version))

    if args.nightly:
        # only keep the latest version
        for key in all_torch_versions[:-1]:
            wheel_info.pop(key)
    elif args.torch_version != 'all':
        torch_versions = args.torch_version.split(',')
        # only keep the torch versions specified
        for key in all_torch_versions:
            if key not in torch_versions:
                wheel_info.pop(key)

    build_colossalai(wheel_info)


if __name__ == '__main__':
    main()
