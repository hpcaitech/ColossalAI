# Code from https://github.com/mit-han-lab/distrifuser/blob/main/scripts/compute_metrics.py
import argparse
import os

import numpy as np
import torch
from cleanfid import fid
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchvision.transforms import Resize
from tqdm import tqdm


def read_image(path: str):
    """
    input: path
    output: tensor (C, H, W)
    """
    img = np.asarray(Image.open(path))
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


class MultiImageDataset(Dataset):
    def __init__(self, root0, root1, is_gt=False):
        super().__init__()
        self.root0 = root0
        self.root1 = root1
        file_names0 = os.listdir(root0)
        file_names1 = os.listdir(root1)

        self.image_names0 = sorted([name for name in file_names0 if name.endswith(".png") or name.endswith(".jpg")])
        self.image_names1 = sorted([name for name in file_names1 if name.endswith(".png") or name.endswith(".jpg")])
        self.is_gt = is_gt
        assert len(self.image_names0) == len(self.image_names1)

    def __len__(self):
        return len(self.image_names0)

    def __getitem__(self, idx):
        img0 = read_image(os.path.join(self.root0, self.image_names0[idx]))
        if self.is_gt:
            # resize to 1024 x 1024
            img0 = Resize((1024, 1024))(img0)
        img1 = read_image(os.path.join(self.root1, self.image_names1[idx]))

        batch_list = [img0, img1]
        return batch_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--is_gt", action="store_true")
    parser.add_argument("--input_root0", type=str, required=True)
    parser.add_argument("--input_root1", type=str, required=True)
    args = parser.parse_args()

    psnr = PeakSignalNoiseRatio(data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3)).to("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda")

    dataset = MultiImageDataset(args.input_root0, args.input_root1, is_gt=args.is_gt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    progress_bar = tqdm(dataloader)
    with torch.inference_mode():
        for i, batch in enumerate(progress_bar):
            batch = [img.to("cuda") / 255 for img in batch]
            batch_size = batch[0].shape[0]
            psnr.update(batch[0], batch[1])
            lpips.update(batch[0], batch[1])
    fid_score = fid.compute_fid(args.input_root0, args.input_root1)

    print("PSNR:", psnr.compute().item())
    print("LPIPS:", lpips.compute().item())
    print("FID:", fid_score)
