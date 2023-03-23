# FastFold Inference

## Table of contents

- [FastFold Inference](#fastfold-inference)
  - [Table of contents](#table-of-contents)
  - [📚 Overview](#-overview)
  - [🚀 Quick Start](#-quick-start)
  - [🔍 Dive into FastFold](#-dive-into-fastfold)

## 📚 Overview

This example lets you to try out the inference of [FastFold](https://github.com/hpcaitech/FastFold).

## 🚀 Quick Start

1. Install FastFold

We highly recommend you to install FastFold with conda.
```
git clone https://github.com/hpcaitech/FastFold
cd FastFold
conda env create --name=fastfold -f environment.yml
conda activate fastfold
python setup.py install
```

2. Download datasets.

It may take ~900GB space to keep datasets.
```
./scripts/download_all_data.sh data/
```

3. Run the inference scripts.

```
bash inference.sh
```
You can find predictions under the `outputs` dir.

## 🔍 Dive into FastFold

There are another features of [FastFold](https://github.com/hpcaitech/FastFold), such as:
+ more excellent kernel based on triton
+ much faster data processing based on ray
+ training supported

More detailed information can be seen [here](https://github.com/hpcaitech/FastFold/).
