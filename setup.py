import os
import sys
from datetime import datetime
from typing import List

from setuptools import find_packages, setup

from op_builder.utils import (
    check_cuda_availability,
    check_pytorch_version,
    check_system_pytorch_cuda_match,
    get_cuda_bare_metal_version,
    get_pytorch_version,
    set_cuda_arch_list,
)

try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_HOME = None

# Some constants for installation checks
MIN_PYTORCH_VERSION_MAJOR = 1
MIN_PYTORCH_VERSION_MINOR = 10
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_CUDA_EXT = int(os.environ.get('CUDA_EXT', '0')) == 1
IS_NIGHTLY = int(os.environ.get('NIGHTLY', '0')) == 1

# a variable to store the op builder
ext_modules = []

# we do not support windows currently
if sys.platform == 'win32':
    raise RuntimeError("Windows is not supported yet. Please try again within the Windows Subsystem for Linux (WSL).")


# check for CUDA extension dependencies
def environment_check_for_cuda_extension_build():
    if not TORCH_AVAILABLE:
        raise ModuleNotFoundError(
            "[extension] PyTorch is not found while CUDA_EXT=1. You need to install PyTorch first in order to build CUDA extensions"
        )

    if not CUDA_HOME:
        raise RuntimeError(
            "[extension] CUDA_HOME is not found while CUDA_EXT=1. You need to export CUDA_HOME environment variable or install CUDA Toolkit first in order to build CUDA extensions"
        )

    check_system_pytorch_cuda_match(CUDA_HOME)
    check_pytorch_version(MIN_PYTORCH_VERSION_MAJOR, MIN_PYTORCH_VERSION_MINOR)
    check_cuda_availability()


def fetch_requirements(path) -> List[str]:
    """
    This function reads the requirements file.

    Args:
        path (str): the path to the requirements file.

    Returns:
        The lines in the requirements file.
    """
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme() -> str:
    """
    This function reads the README.md file in the current directory.

    Returns:
        The lines in the README file.
    """
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def get_version() -> str:
    """
    This function reads the version.txt and generates the colossalai/version.py file.

    Returns:
        The library version stored in version.txt.
    """

    setup_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(setup_file_path)
    version_txt_path = os.path.join(project_path, 'version.txt')
    version_py_path = os.path.join(project_path, 'colossalai/version.py')

    with open(version_txt_path) as f:
        version = f.read().strip()

    # write version into version.py
    with open(version_py_path, 'w') as f:
        f.write(f"__version__ = '{version}'\n")

        # look for pytorch and cuda version
        if BUILD_CUDA_EXT:
            torch_major, torch_minor, _ = get_pytorch_version()
            torch_version = f'{torch_major}.{torch_minor}'
            cuda_version = '.'.join(get_cuda_bare_metal_version(CUDA_HOME))
        else:
            torch_version = None
            cuda_version = None

        # write the version into the python file
        if torch_version:
            f.write(f'torch = "{torch_version}"\n')
        else:
            f.write('torch = None\n')

        if cuda_version:
            f.write(f'cuda = "{cuda_version}"\n')
        else:
            f.write('cuda = None\n')

    return version


if BUILD_CUDA_EXT:
    environment_check_for_cuda_extension_build()
    set_cuda_arch_list(CUDA_HOME)

    from op_builder import ALL_OPS
    op_names = []

    # load all builders
    for name, builder_cls in ALL_OPS.items():
        op_names.append(name)
        ext_modules.append(builder_cls().builder())

    # show log
    op_name_list = ', '.join(op_names)
    print(f"[extension]  loaded builders for {op_name_list}")

# always put not nightly branch as the if branch
# otherwise github will treat colossalai-nightly as the project name
# and it will mess up with the dependency graph insights
if not IS_NIGHTLY:
    version = get_version()
    package_name = 'colossalai'
else:
    # use date as the nightly version
    version = datetime.today().strftime('%Y.%m.%d')
    package_name = 'colossalai-nightly'

setup(name=package_name,
      version=version,
      packages=find_packages(exclude=(
          'op_builder',
          'benchmark',
          'docker',
          'tests',
          'docs',
          'examples',
          'tests',
          'scripts',
          'requirements',
          '*.egg-info',
      )),
      description='An integrated large-scale model training system with efficient parallelization techniques',
      long_description=fetch_readme(),
      long_description_content_type='text/markdown',
      license='Apache Software License 2.0',
      url='https://www.colossalai.org',
      project_urls={
          'Forum': 'https://github.com/hpcaitech/ColossalAI/discussions',
          'Bug Tracker': 'https://github.com/hpcaitech/ColossalAI/issues',
          'Examples': 'https://github.com/hpcaitech/ColossalAI-Examples',
          'Documentation': 'http://colossalai.readthedocs.io',
          'Github': 'https://github.com/hpcaitech/ColossalAI',
      },
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension} if ext_modules else {},
      install_requires=fetch_requirements('requirements/requirements.txt'),
      entry_points='''
        [console_scripts]
        colossalai=colossalai.cli:cli
    ''',
      python_requires='>=3.6',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: Apache Software License',
          'Environment :: GPU :: NVIDIA CUDA',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: System :: Distributed Computing',
      ],
      package_data={
          'colossalai': [
              '_C/*.pyi', 'kernel/cuda_native/csrc/*', 'kernel/cuda_native/csrc/kernel/*',
              'kernel/cuda_native/csrc/kernels/include/*'
          ]
      })
