import json
import os
import tempfile
from typing import Optional

import pytest
import torch
from coati.dataset.prompt_dataset import PromptDataset
from coati.dataset.reward_dataset import PreferenceDataset
from coati.dataset.sft_dataset import IGNORE_INDEX, SFTDataset, SupervisedDataset
from coati.models.chatglm.chatglm_tokenizer import ChatGLMTokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, BloomTokenizerFast, LlamaTokenizer, PreTrainedTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

SFT_DATASET = [
    {
        "instruction": "Provide a list of the top 10 most popular mobile games in Asia",
        "input": "",
        "output": "The top 10 most popular mobile games in Asia are:\n1) PUBG Mobile\n2) Pokemon Go\n3) Candy Crush Saga\n4) Free Fire\n5) Clash of Clans\n6) Mario Kart Tour\n7) Arena of Valor\n8) Fantasy Westward Journey\n9) Subway Surfers\n10) ARK Survival Evolved",
        "id": 0,
    },
    {
        "instruction": "Please provide an action plan for reducing carbon footprint on a corporate level",
        "input": "",
        "output": "An action plan for reducing carbon footprint on a corporate level could include switching to renewable energy sources (e.g., solar or wind powered), investing in energy efficient infrastructure, implementing green production practices (e.g., reduce water and paper usage, dispose of materials responsibly, recycle/reuse materials), reducing or eliminating travel, encouraging employees to be conscious of and reduce their own energy usage, and incentivizing employees to participate in sustainable work practices.",
        "id": 1,
    },
    {
        "instruction": "Write a persuasive email to your boss explaining why you should have a pay raise",
        "input": "",
        "output": "Dear [Boss], \n\nI hope this message finds you well. I am writing to request a pay raise.\n\nAs you know, I have been a dedicated and hardworking member of the team since I started working here [insert number] of months/years ago. My enthusiasm and passion for my job has remained consistent over the years, and I have always given 100% to my role. \n\nI understand that the current financial situation is challenging, however, I would sincerely appreciate you taking the time to consider my request. I believe that my dedication to the job and the value that I bring to the organization warrants a raise. I work diligently and am confident that I can continue to be an asset to the company. \n\nI hope my request is taken into account and I thank you in advance for your understanding. I look forward to our conversation. \n\nSincerely,\n[Your Name]",
        "id": 2,
    },
]

PROMPT_DATASET = [
    {
        "instruction": 'Edit this paragraph to make it more concise: "Yesterday, I went to the store and bought some things. Then, I came home and put them away. After that, I went for a walk and met some friends."',
        "id": 0,
    },
    {"instruction": "Write a descriptive paragraph about a memorable vacation you went on", "id": 1},
    {"instruction": "Write a persuasive essay arguing why homework should be banned in schools", "id": 2},
    {"instruction": "Create a chart comparing the statistics on student debt in the United States.", "id": 3},
]


def make_tokenizer(model: str):
    if model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    elif model == "bloom":
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
        tokenizer.pad_token = tokenizer.eos_token
    elif model == "opt":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        tokenizer.pad_token = tokenizer.eos_token
    elif model == "llama":
        tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        tokenizer.pad_token = tokenizer.unk_token
    elif model == "chatglm":
        tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported model '{model}'")
    return tokenizer


def check_content(input_ids_stripped: torch.Tensor, tokenizer: PreTrainedTokenizer, model: str):
    if model == "opt":
        # NOTE:  Contrary to GPT2, OPT adds the EOS token </s> to the beginning of every prompt.
        assert input_ids_stripped[0] == tokenizer.eos_token_id
        input_ids_stripped = input_ids_stripped[1:]
    elif model == "llama":
        assert input_ids_stripped[0] == tokenizer.bos_token_id
        input_ids_stripped = input_ids_stripped[1:]
    elif model == "chatglm":
        assert input_ids_stripped[0] == tokenizer.bos_token_id
        assert input_ids_stripped[-1] == tokenizer.eos_token_id
        input_ids_stripped = input_ids_stripped[1:-1]
    assert torch.all(input_ids_stripped != tokenizer.pad_token_id)
    assert torch.all(input_ids_stripped != tokenizer.bos_token_id)
    assert torch.all(input_ids_stripped != tokenizer.eos_token_id)
    assert input_ids_stripped != tokenizer.sep_token_id
    assert input_ids_stripped != tokenizer.cls_token_id
    if model == "chatglm":
        assert torch.all(input_ids_stripped != tokenizer.mask_token_id)
    else:
        assert input_ids_stripped != tokenizer.mask_token_id


@pytest.mark.parametrize("model", ["gpt2", "bloom", "opt", "llama"])
@pytest.mark.parametrize("max_length", [32, 1024])
@pytest.mark.parametrize("max_datasets_size", [2])
def test_prompt_dataset(model: str, max_datasets_size: int, max_length: int):
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_name = "prompt_dataset.json"
        with open(os.path.join(tmp_dir, dataset_name), "w") as f:
            json.dump(PROMPT_DATASET, f)
        tokenizer = make_tokenizer(model)
        assert tokenizer.padding_side in ("left", "right")
        prompt_dataset = PromptDataset(
            data_path=os.path.join(tmp_dir, dataset_name),
            tokenizer=tokenizer,
            max_datasets_size=max_datasets_size,
            max_length=max_length,
        )
        assert len(prompt_dataset) == min(max_datasets_size, len(PROMPT_DATASET))
        for i in range(len(prompt_dataset)):
            assert isinstance(prompt_dataset[i], dict)
            assert list(prompt_dataset[i].keys()) == ["input_ids", "attention_mask"]
            input_ids = prompt_dataset[i]["input_ids"]
            attention_mask = prompt_dataset[i]["attention_mask"]
            attention_mask = attention_mask.bool()
            assert input_ids.shape == attention_mask.shape == torch.Size([max_length])
            assert torch.all(input_ids[torch.logical_not(attention_mask)] == tokenizer.pad_token_id)
            check_content(input_ids.masked_select(attention_mask), tokenizer, model)


@pytest.mark.parametrize("model", ["gpt2", "bloom", "opt", "llama"])
@pytest.mark.parametrize(
    ["dataset_path", "subset"], [("Anthropic/hh-rlhf", "harmless-base"), ("Dahoas/rm-static", None)]
)
@pytest.mark.parametrize("max_datasets_size", [32])
@pytest.mark.parametrize("max_length", [32, 1024])
def test_reward_dataset(model: str, dataset_path: str, subset: Optional[str], max_datasets_size: int, max_length: int):
    data = load_dataset(dataset_path, data_dir=subset)
    assert max_datasets_size <= len(data["train"]) and max_datasets_size <= len(data["test"])
    train_data = data["train"].select(range(max_datasets_size))
    test_data = data["test"].select(range(max_datasets_size))
    tokenizer = make_tokenizer(model)
    assert tokenizer.padding_side in ("left", "right")

    if dataset_path == "Anthropic/hh-rlhf":
        train_dataset = PreferenceDataset(train_data, tokenizer, max_length)
        test_dataset = PreferenceDataset(test_data, tokenizer, max_length)
    elif dataset_path == "Dahoas/rm-static":
        train_dataset = PreferenceDataset(
            train_data,
            tokenizer,
            max_length,
            dataset_schema={"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        )
        test_dataset = PreferenceDataset(
            test_data,
            tokenizer,
            max_length,
            dataset_schema={"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        )
    else:
        raise ValueError(f'Unsupported dataset "{dataset_path}"')

    assert len(train_dataset) == len(test_dataset) == max_datasets_size
    for i in range(max_datasets_size):
        chosen_ids, c_mask, reject_ids, r_mask = train_dataset[i]
        assert chosen_ids.shape == c_mask.shape == reject_ids.shape == r_mask.shape == torch.Size([max_length])
        c_mask = c_mask.to(torch.bool)
        r_mask = r_mask.to(torch.bool)
        if chosen_ids.masked_select(c_mask)[-1] == tokenizer.eos_token_id:
            check_content(chosen_ids.masked_select(c_mask)[:-1], tokenizer, model)
            assert torch.all(chosen_ids.masked_select(torch.logical_not(c_mask)) == tokenizer.pad_token_id)
        else:
            check_content(chosen_ids.masked_select(c_mask), tokenizer, model)
            assert torch.all(c_mask)
        if reject_ids.masked_select(r_mask)[-1] == tokenizer.eos_token_id:
            check_content(reject_ids.masked_select(r_mask)[:-1], tokenizer, model)
            assert torch.all(reject_ids.masked_select(torch.logical_not(r_mask)) == tokenizer.pad_token_id)
        else:
            check_content(reject_ids.masked_select(r_mask), tokenizer, model)
            assert torch.all(r_mask)

        chosen_ids, c_mask, reject_ids, r_mask = test_dataset[i]
        assert chosen_ids.shape == c_mask.shape == reject_ids.shape == r_mask.shape == torch.Size([max_length])
        c_mask = c_mask.to(torch.bool)
        r_mask = r_mask.to(torch.bool)
        if chosen_ids.masked_select(c_mask)[-1] == tokenizer.eos_token_id:
            check_content(chosen_ids.masked_select(c_mask)[:-1], tokenizer, model)
            assert torch.all(chosen_ids.masked_select(torch.logical_not(c_mask)) == tokenizer.pad_token_id)
        else:
            check_content(chosen_ids.masked_select(c_mask), tokenizer, model)
            assert torch.all(c_mask)
        if reject_ids.masked_select(r_mask)[-1] == tokenizer.eos_token_id:
            check_content(reject_ids.masked_select(r_mask)[:-1], tokenizer, model)
            assert torch.all(reject_ids.masked_select(torch.logical_not(r_mask)) == tokenizer.pad_token_id)
        else:
            check_content(reject_ids.masked_select(r_mask), tokenizer, model)
            assert torch.all(r_mask)


@pytest.mark.parametrize("model", ["gpt2", "bloom", "opt", "llama"])  # temperally disable test for chatglm
@pytest.mark.parametrize("dataset_path", ["yizhongw/self_instruct", None])
@pytest.mark.parametrize("max_dataset_size", [2])
@pytest.mark.parametrize("max_length", [32, 1024])
def test_sft_dataset(model: str, dataset_path: Optional[str], max_dataset_size: int, max_length: int):
    tokenizer = make_tokenizer(model)
    if dataset_path == "yizhongw/self_instruct":
        data = load_dataset(dataset_path, "super_natural_instructions")
        train_data = data["train"].select(range(max_dataset_size))
        sft_dataset = SFTDataset(train_data, tokenizer, max_length)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_name = "sft_dataset.json"
            with open(os.path.join(tmp_dir, dataset_name), "w") as f:
                json.dump(SFT_DATASET, f)
            sft_dataset = SupervisedDataset(
                tokenizer=tokenizer,
                data_path=os.path.join(tmp_dir, dataset_name),
                max_datasets_size=max_dataset_size,
                max_length=max_length,
            )
        assert len(sft_dataset) == min(max_dataset_size, len(SFT_DATASET))

    if isinstance(tokenizer, ChatGLMTokenizer):
        for i in range(max_dataset_size):
            assert isinstance(sft_dataset[i], dict)
            assert list(sft_dataset[i].keys()) == ["input_ids", "labels"]
            input_ids = sft_dataset[i]["input_ids"]
            labels = sft_dataset[i]["labels"]
            assert input_ids.shape == labels.shape == torch.Size([max_length])

            ignore_mask = labels == IGNORE_INDEX
            assert input_ids.masked_select(torch.logical_not(ignore_mask))[0] == tokenizer.bos_token_id
            check_content(input_ids.masked_select(torch.logical_not(ignore_mask)), tokenizer, model)
            return

    for i in range(max_dataset_size):
        assert isinstance(sft_dataset[i], dict)
        assert list(sft_dataset[i].keys()) == ["input_ids", "labels", "attention_mask"]
        input_ids = sft_dataset[i]["input_ids"]
        labels = sft_dataset[i]["labels"]
        attention_mask = sft_dataset[i]["attention_mask"].to(torch.bool)
        assert input_ids.shape == labels.shape == attention_mask.shape == torch.Size([max_length])
        if input_ids.masked_select(attention_mask)[-1] == tokenizer.eos_token_id:
            check_content(input_ids.masked_select(attention_mask)[:-1], tokenizer, model)
            assert torch.all(input_ids.masked_select(torch.logical_not(attention_mask)) == tokenizer.pad_token_id)
        else:
            check_content(input_ids.masked_select(attention_mask), tokenizer, model)
            assert torch.all(attention_mask)
        ignore_mask = labels == IGNORE_INDEX
        prompt_mask = torch.logical_and(ignore_mask, attention_mask)
        check_content(input_ids.masked_select(prompt_mask), tokenizer, model)
        assert torch.all(input_ids.masked_select(ignore_mask ^ prompt_mask) == tokenizer.pad_token_id)


if __name__ == "__main__":
    test_sft_dataset(model="bloom", dataset_path="yizhongw/self_instruct", max_dataset_size=2, max_length=256)
    test_sft_dataset(model="bloom", dataset_path="custom", max_dataset_size=2, max_length=256)

    test_reward_dataset(
        model="gpt2", dataset_path="Anthropic/hh-rlhf", subset="harmless-base", max_datasets_size=8, max_length=256
    )

    test_prompt_dataset(model="opt", max_datasets_size=2, max_length=128)
