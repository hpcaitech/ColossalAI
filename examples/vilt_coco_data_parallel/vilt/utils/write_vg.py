import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions):
    name = path.split("/")[-1]
    iid = int(name[:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    cdicts = iid2captions[iid]
    captions = [c["phrase"] for c in cdicts]
    widths = [c["width"] for c in cdicts]
    heights = [c["height"] for c in cdicts]
    xs = [c["x"] for c in cdicts]
    ys = [c["y"] for c in cdicts]

    return [
        binary,
        captions,
        widths,
        heights,
        xs,
        ys,
        str(iid),
    ]


def make_arrow(root, dataset_root):
    with open(f"{root}/annotations/region_descriptions.json", "r") as fp:
        captions = json.load(fp)

    iid2captions = defaultdict(list)
    for cap in tqdm(captions):
        cap = cap["regions"]
        for c in cap:
            iid2captions[c["image_id"]].append(c)

    paths = list(glob(f"{root}/images/VG_100K/*.jpg")) + list(
        glob(f"{root}/images/VG_100K_2/*.jpg")
    )
    random.shuffle(paths)
    caption_paths = [
        path for path in paths if int(path.split("/")[-1][:-4]) in iid2captions
    ]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions) for path in tqdm(caption_paths)]
    dataframe = pd.DataFrame(
        bs, columns=["image", "caption", "width", "height", "x", "y", "image_id"],
    )
    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/vg.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
