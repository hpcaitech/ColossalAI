import os
import re
import sys
from typing import List

from setuptools import find_namespace_packages, setup

try:
    import torch  # noqa
    from torch.utils.cpp_extension import BuildExtension

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_EXT = int(os.environ.get("BUILD_EXT", "0")) == 1

# we do not support windows currently
if sys.platform == "win32":
    raise RuntimeError("Windows is not supported yet. Please try again within the Windows Subsystem for Linux (WSL).")


def fetch_requirements(path) -> List[str]:
    """
    This function reads the requirements file.

    Args:
        path (str): the path to the requirements file.

    Returns:
        The lines in the requirements file.
    """
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def get_install_requirements(path, torch_version=None) -> List[str]:
    """Pin binary builds to their Torch public version; preserve JIT requirements."""
    requirements = fetch_requirements(path)
    if torch_version is None:
        return requirements
    # CUDA flavor is validated separately; a public version pin cannot encode it.
    pinned = f"torch=={str(torch_version).split('+', 1)[0]}"
    result = []
    for requirement in requirements:
        if re.match(r"^torch(?=[\s<>=!~;\[]|$)", requirement, re.IGNORECASE):
            if pinned not in result:
                result.append(pinned)
        else:
            result.append(requirement)
    if pinned not in result:
        result.append(pinned)
    return result


def fetch_readme() -> str:
    """
    This function reads the README.md file in the current directory.

    Returns:
        The lines in the README file.
    """
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def get_version() -> str:
    """
    This function reads the version.txt and generates the colossalai/version.py file.

    Returns:
        The library version stored in version.txt.
    """

    setup_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(setup_file_path)
    version_txt_path = os.path.join(project_path, "version.txt")
    version_py_path = os.path.join(project_path, "colossalai/version.py")

    with open(version_txt_path) as f:
        version = f.read().strip()

    # write version into version.py
    with open(version_py_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
    return version


if BUILD_EXT:
    if not TORCH_AVAILABLE:
        raise ModuleNotFoundError(
            "[extension] PyTorch is not found while BUILD_EXT=1. You need to install PyTorch first in order to build CUDA extensions"
        )

    from extensions import ALL_EXTENSIONS

    op_names = []
    ext_modules = []

    for ext_cls in ALL_EXTENSIONS:
        ext = ext_cls()
        if ext.support_aot and ext.is_available():
            ext.assert_build_compatible()
            op_names.append(ext.name)
            ext_modules.append(ext.build_aot())

    # show log
    if len(ext_modules) == 0:
        raise RuntimeError("[extension] Could not find any kernel compatible with the current environment.")
    else:
        op_name_list = ", ".join(op_names)
        print(f"[extension] Building extensions{op_name_list}")
else:
    ext_modules = []

version = get_version()
package_name = "colossalai"

setup(
    name=package_name,
    version=version,
    # Some public modules, including autochunk and legacy.moe, use namespace packages.
    packages=find_namespace_packages(include=("colossalai", "colossalai.*"), exclude=("*.__pycache__",)),
    description="An integrated large-scale model training system with efficient parallelization techniques",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    url="https://www.colossalai.org",
    project_urls={
        "Forum": "https://github.com/hpcaitech/ColossalAI/discussions",
        "Bug Tracker": "https://github.com/hpcaitech/ColossalAI/issues",
        "Examples": "https://github.com/hpcaitech/ColossalAI-Examples",
        "Documentation": "http://colossalai.readthedocs.io",
        "Github": "https://github.com/hpcaitech/ColossalAI",
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    install_requires=get_install_requirements(
        "requirements/requirements.txt", torch.__version__ if BUILD_EXT else None
    ),
    entry_points="""
        [console_scripts]
        colossalai=colossalai.cli:cli
    """,
    python_requires=">=3.10,<3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    package_data={
        "colossalai": [
            "kernel/extensions/csrc/**/*",
            "kernel/extensions/pybind/**/*",
        ]
    },
)
