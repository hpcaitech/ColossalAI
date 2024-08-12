import argparse
import json
import os

sft_seed = {
    "messages": [
        {"from": "user", "content": "Give three tips for staying healthy."},
        {
            "from": "assistant",
            "content": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.",
        },
    ]
}
prompt_seed = {
    "messages": [
        {"from": "user", "content": "Describe the impacts of climate change on communities living in coastal areas."},
        {
            "from": "assistant",
            "content": "Climate change has caused an increase in sea levels, which has caused coastal erosion and flooding of low-lying areas. This has led to displacement of people from their homes, as well as increased risk of epidemics of waterborne illnesses. Coastal cities have also seen an increase in extreme weather events such as hurricanes and tropical storms, which can cause extensive damage to infrastructure, homes, and businesses. As a result of climate change, some coastal areas are becoming uninhabitable, forcing communities to seek alternative living arrangements.",
        },
    ]
}
preference_seed = {
    "context": [
        {"from": "user", "content": "What kind of noises did dinosaurs make?"},
        {
            "from": "assistant",
            "content": "Humans and dinosaurs didn't live at the same time, so it's really hard to say. The best place to find out what noises dinosaurs made would be",
        },
        {"from": "user", "content": "yes they did"},
        {
            "from": "assistant",
            "content": "to guess, and that would probably require lots of reading and a certain amount of imagination, so we're not really prepared to do that.",
        },
        {"from": "user", "content": "you cant read"},
    ],
    "chosen": [{"from": "assistant", "content": "You can read?"}],
    "rejected": [{"from": "assistant", "content": "there's a lot of stuff humans don't know"}],
}
kto_seed = {
    "prompt": [
        {"from": "user", "content": "What are some praise words in english?"},
        {
            "from": "assistant",
            "content": "Here's an incomplete list.\n\nexcellent, fantastic, impressive  ...",
        },
        {"from": "user", "content": "What's your favorite one?"},
    ],
    "completion": {"from": "assistant", "content": "Impressive."},
    "label": True,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        default=None,
        help="The output dir",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        default=None,
        help="The type of data",
    )
    args = parser.parse_args()
    if args.data_type == "sft":
        seed = sft_seed
    elif args.data_type == "prompt":
        seed = prompt_seed
    elif args.data_type == "preference":
        seed = preference_seed
    elif args.data_type == "kto":
        seed = kto_seed
    else:
        raise ValueError(f"Unknown data type {args.data_type}")
    if args.data_type != "kto":
        line = json.dumps(seed, ensure_ascii=False) + "\n"
        for idx in [1, 2, 3]:
            with open(os.path.join(args.data_dir, f"{idx}.jsonl"), "w", encoding="utf8") as f:
                for i in range(1000):
                    f.write(line)
                f.write(line)
    else:
        for idx in [1, 2, 3]:
            with open(os.path.join(args.data_dir, f"{idx}.jsonl"), "w", encoding="utf8") as f:
                for i in range(1000):
                    seed["label"] = not seed["label"]
                    line = json.dumps(seed, ensure_ascii=False) + "\n"
                    f.write(line)
