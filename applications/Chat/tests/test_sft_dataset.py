import argparse
import pytest
import torch
from datasets import load_dataset
from coati.dataset import DataCollatorForSupervisedDataset, SFTDataset, SupervisedDataset
from transformers import AutoTokenizer


DATASET = [
        {"instruction": "Reorder the following list in a chronological order.", \
            "input": "Event A, Event B, Event C, Event D", \
            "output": "Event A, Event B, Event C, Event D \n\nChronological Order: Event A, Event B, Event C, Event D"}, \
        {"instruction": "Find a specific example of the given keyword.", \
            "input": "Keyword: Humanitarian efforts", \
            "output": "The World Food Programme is an example of a humanitarian effort, as it provides food aid to individuals in conflict-affected and hunger-stricken areas."}
        ]


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Testing SFT Dataset")
    parser.add_argument('--data', type=str, required=True, help="Path to train data fro testing SupervisedDataset class")

    return parser.parse_args()


def test_Dataset(train_dataset):
    '''test SFTDataset class and SupervisedDataset class
    '''
    # the type of train_dataset.input_ids and train_dataset.labels should be list
    assert isinstance(train_dataset.input_ids, list)
    assert isinstance(train_dataset.labels, list)
    # the length of train_dataset should >= 0
    assert len(train_dataset)>=0
    # input ids and labels should be the same
    for i in range(len(train_dataset)):
        assert torch.equal(train_dataset.input_ids[i], train_dataset.input_ids[i])

    # check __getitem__
    item = train_dataset[0]
    assert isinstance(item, dict)
    assert list(item.keys()) == ['input_ids', 'labels']

    return


def test_collator(data_collator, train_dataset):
    ''' test DataCollatorForSupervisedDataset class
    '''
    data = data_collator(train_dataset)
    assert isinstance(data, dict)
    assert list(data.keys()) == ['input_ids', 'labels', 'attention_mask']

    # dtype of attention_mask should be Boolean
    assert data['attention_mask'].dtype == torch.bool


@pytest.mark.dataset
def test_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    train_data_sft = load_dataset('yizhongw/self_instruct', 'super_natural_instructions', split='train')
    train_dataset_sft = SFTDataset(train_data_sft, tokenizer, max_length=16)

    # test SFTDataset
    test_Dataset(train_dataset_sft)

    train_dataset_spv = SupervisedDataset(tokenizer=tokenizer,
                                          data_path=dataset,
                                          max_datasets_size=16,
                                          max_length=16)
    
    # test SupervisedDataset
    test_Dataset(train_dataset_spv)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # test DataCollatorForSupervisedDataset
    test_collator(data_collator, train_dataset_spv)


if __name__ == '__main__':
    args = parseArgs()
    test_dataset(args.data)