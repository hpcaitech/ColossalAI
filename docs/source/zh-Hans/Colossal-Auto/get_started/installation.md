# 安装

## 声明

我们的自动并行功能处于alpha版本，仍在快速的开发迭代中。我们会在兼容性和稳定性上做持续地改进。如果您遇到任何问题，欢迎随时提issue给我们。


## 要求

我们需要一些额外的依赖性来支持自动并行功能。 请在使用自动平行之前安装它们。

### 安装PyTorch

我们仅支持Pytorch 1.12，现在未测试其他版本。 将来我们将支持更多版本。

```bash
#conda
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
#pip
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### 安装pulp和coin-or-cbc

```bash
pip install pulp
conda install -c conda-forge coin-or-cbc
```
