import requests
from bs4 import BeautifulSoup
import re
import os
import subprocess


WHEEL_TEXT_ROOT_URL = 'https://github.com/hpcaitech/public_assets/tree/main/colossalai/torch_wheel/wheel_urls'
RAW_TEXT_FILE_PREFIX = 'https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/torch_wheel/wheel_urls'
CUDA_HOME = os.environ['CUDA_HOME']

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
        if 'cu' in a_link.text and '.txt' in a_link.text:
            filename = a_link.text
            torch_version, cuda_version = filename.rstrip('.txt').split('-')

            wheel_info[torch_version] = dict()
            wheel_info[torch_version][cuda_version] = dict()

            file_text = requests.get(f'{RAW_TEXT_FILE_PREFIX}/{filename}').text
            wheel_urls = file_text.strip().split('\n')

            for url in wheel_urls:
                for part in url.split('-'):
                    if re.search('cp\d+', part):
                        python_version = f"3.{part.lstrip('cp3')}"
                        wheel_info[torch_version][cuda_version][python_version] = url
                        break
    return wheel_info


def build_colossalai(wheel_info):
    cuda_version_major, cuda_version_minor = get_cuda_bare_metal_version()
    cuda_version_on_host = f'cu{cuda_version_major}{cuda_version_minor}'

    for torch_version, cuda_versioned_wheel_info in wheel_info.items():
        for cuda_version, python_versioned_wheel_info in cuda_versioned_wheel_info.items():
            if cuda_version_on_host == cuda_version:
                for python_version, torch_wheel_url in python_versioned_wheel_info.items():
                    filename = torch_wheel_url.split('/')[-1]
                    cmd = f'bash ./build_colossalai_wheel.sh {python_version} {torch_wheel_url} {filename}'
                    os.system(cmd)

def main():
    wheel_info = all_wheel_info()
    build_colossalai(wheel_info)

if __name__ == '__main__':
    main()





        