from setuptools import find_packages, setup


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def fetch_version():
    with open("version.txt", "r") as f:
        return f.read().strip()


print(find_packages(exclude=("tests", "*.egg-info", "data", "examples")))
setup(
    name="colossalqa",
    version=fetch_version(),
    packages=find_packages(exclude=("tests", "*.egg-info", "data", "examples")),
    description="Colossal-AI powered retrieval QA",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    url="https://github.com/hpcaitech/Coati",
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
