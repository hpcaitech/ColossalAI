# 快速安装

## 使用pip安装

```bash
pip install colossalai
```

## 使用源代码安装

```shell
git clone git@github.com:hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

安装并支持内核融合（使用融合优化器时必须执行下面的代码）

```
pip install -v --no-cache-dir --global-option="--cuda_ext" .
```
