from setuptools import setup, find_packages

setup(
    name="vilt",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]
    ),
    version="1.0.0",
    license="MIT",
    description="ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision",
    author="Wonjae Kim",
    author_email="wonjaekim@kakao.com",
    url="https://github.com/dandelin/vilt'",
    keywords=["vision and language pretraining"],
    install_requires=["torch", "pytorch_lightning"],
)
