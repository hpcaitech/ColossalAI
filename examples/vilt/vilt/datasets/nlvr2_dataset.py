from .base_dataset import BaseDataset
import sys
import random


class NLVR2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["nlvr2_train"]
        elif split == "val":
            names = ["nlvr2_dev", "nlvr2_test1"]
        elif split == "test":
            names = ["nlvr2_dev", "nlvr2_test1"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        result = None
        while result is None:
            try:
                image_tensor_0 = self.get_image(index, image_key="image_0")["image"]
                image_tensor_1 = self.get_image(index, image_key="image_1")["image"]
                text = self.get_text(index)["text"]
                result = True
            except:
                print(
                    f"error while read file idx {index} in {self.names[0]}",
                    file=sys.stderr,
                )
                index = random.randint(0, len(self.index_mapper) - 1)

        index, question_index = self.index_mapper[index]
        answers = self.table["answers"][index][question_index].as_py()
        answers = answers == "True"

        return {
            "image_0": image_tensor_0,
            "image_1": image_tensor_1,
            "text": text,
            "answers": answers,
            "table_name": self.table_names[index],
        }
