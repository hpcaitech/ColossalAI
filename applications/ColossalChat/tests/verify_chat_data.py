import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_source",
        type=str,
        required=True,
        default=None,
        help="The raw data file",
    )
    parser.add_argument(
        "--to_verify_file",
        type=str,
        required=True,
        default=None,
        help="The file that contains the data to be verified",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        default=None,
        help="The data type",
    )
    args = parser.parse_args()

    # Read data
    data = []
    with open(args.data_source, "r", encoding="utf8") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    to_verify_data = []
    with open(args.to_verify_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            to_verify_data.append(json.loads(line))

    if args.data_type == "sft":
        target_lable = [msg["content"].strip() for msg in data[0]["messages"] if msg["from"] == "assistant"]
        target_negative_label = [msg["content"].strip() for msg in data[0]["messages"] if msg["from"] == "human"]

        # Read to verify file

        to_verify_lable = to_verify_data[0]["labels_decode"]
        for label in target_lable:
            assert any([label in s for s in to_verify_lable]), f"Label {label} not in target label {to_verify_lable}"
        for label in target_negative_label:
            assert all(
                [label not in s for s in to_verify_lable]
            ), f"Negative label {label} in target label {to_verify_lable}"
    elif args.data_type == "dpo":
        chosen_lable = data[0]["chosen"][0]["content"].strip()
        rejected_lable = data[0]["rejected"][0]["content"].strip()

        # Read to verify file
        to_verify_lable_chosen = to_verify_data[0]["chosen_label_decode"]
        to_verify_lable_rejected = to_verify_data[0]["rejected_label_decode"]
        assert any(
            [chosen_lable in s for s in to_verify_lable_chosen]
        ), f"Chosen label {chosen_lable} not in target chosen label {to_verify_lable_chosen}"
        assert any(
            [rejected_lable in s for s in to_verify_lable_rejected]
        ), f"Rejected label {rejected_lable} not in target rejected label {to_verify_lable_chosen}"
    elif args.data_type == "kto":
        sample = data[0]
        to_verify_data = to_verify_data[0]
        for line in sample["prompt"]:
            assert line["content"] in to_verify_data["input_id_decode"]
        assert sample["completion"]["content"] in to_verify_data["input_id_decode"]
        assert sample["completion"]["content"] in to_verify_data["completion_decode"]
        assert sample["label"] == to_verify_data["label"]
