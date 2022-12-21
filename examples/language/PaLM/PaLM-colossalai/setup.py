from setuptools import find_packages, setup

setup(
    name="PaLM-pytorch",
    packages=find_packages(exclude=[]),
    version="0.2.2",
    license="MIT",
    description="PaLM: Scaling Language Modeling with Pathways - Pytorch",
    author="Phil Wang",
    author_email="lucidrains@gmail.com",
    long_description_content_type = 'text/markdown',
    url="https://github.com/lucidrains/PaLM-pytorch",
    keywords=[
        "artificial general intelligence",
        "deep learning",
        "transformers",
        "attention mechanism",
    ],
    install_requires=[
        "einops>=0.4",
        "torch>=1.6",
        "triton>=2.0dev"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
