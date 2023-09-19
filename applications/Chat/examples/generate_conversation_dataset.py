import argparse
import json

from datasets import load_dataset


def generate_alpaca():
    # We can convert dataset with the same format("instruction", "input", "output") as Alpaca into a one-round conversation.
    conversation_dataset = []
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    instructions = dataset["instruction"]
    inputs = dataset["input"]
    outputs = dataset["output"]

    assert len(instructions) == len(inputs) == len(outputs)

    for idx in range(len(instructions)):
        human_utterance = instructions[idx] + "\n\n" + inputs[idx] if inputs[idx] else instructions[idx]
        human = {"from": "human", "value": human_utterance}

        gpt_utterance = outputs[idx]
        gpt = {"from": "gpt", "value": gpt_utterance}

        conversation = dict(type="instruction", language="English", dataset="Alpaca", conversations=[human, gpt])
        conversation_dataset.append(conversation)

    return conversation_dataset


def generate_sharegpt():
    # ShareGPT data requires less processing.
    conversation_dataset = []
    dataset = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
        split="train",
    )

    conversations = dataset["conversations"]

    for idx in range(len(conversations)):
        for conv in conversations[idx]:
            # We don't need markdown and text value.
            del conv["markdown"]
            del conv["text"]

        conversation = dict(
            type="conversation", language="Multilingual", dataset="ShareGPT", conversations=conversations[idx]
        )
        conversation_dataset.append(conversation)

    return conversation_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="All",
        choices=["Alpaca", "ShareGPT", "All"],
        help="which dataset to convert, All will combine Alpaca and ShareGPT",
    )
    parser.add_argument("--save_path", type=str, default="dataset.json", help="path to save the converted dataset")
    args = parser.parse_args()

    conversation_dataset = []

    if args.dataset == "Alpaca":
        conversation_dataset.extend(generate_alpaca())
    elif args.dataset == "ShareGPT":
        conversation_dataset.extend(generate_sharegpt())
    else:
        conversation_dataset.extend(generate_alpaca())
        conversation_dataset.extend(generate_sharegpt())

    for idx, sample in enumerate(conversation_dataset):
        sample["id"] = idx + 1

    with open(args.save_path, mode="w") as f:
        json.dump(conversation_dataset, f, indent=4, default=str, ensure_ascii=False)
