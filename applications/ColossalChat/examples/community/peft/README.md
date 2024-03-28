:warning: **This content may be outdated since the major update of Colossal Chat. We will update this content soon.**

# Add Peft support for SFT and Prompts model training

The original implementation just adopts the loralib and merges the layers into the final model. The huggingface peft is a better lora model implementation and can be easily training and distributed.

Since reward model is relative small, I just keep it as original one. I suggest train full model to get the proper reward/critic model.

# Preliminary installation

Since the current pypi peft package(0.2) has some bugs, please install the peft package using source.

```
git clone https://github.com/huggingface/peft
cd peft
pip install .
```

# Usage

For SFT training, just call train_peft_sft.py

Its arguments are almost identical to train_sft.py instead adding a new eval_dataset if you have an eval_dataset file. The data file is just a plain datafile, please check the format in the easy_dataset.py.

For stage-3 rlhf training, call train_peft_prompts.py.
Its arguments are almost identical to train_prompts.py. The only difference is that I use text files to indicate the prompt and pretrained data file. The models are included in easy_models.py. Currently only bloom models are tested, but technically gpt2/opt/llama should be supported.

# Dataformat

Please refer the formats in test_sft.txt, test_prompts.txt, test_pretrained.txt.
