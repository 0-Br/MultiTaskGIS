import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


all_classes = {
    0: "no-data",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture"}


data_transforms = {
    "train": transforms.Compose([
        transforms.Normalize([0.277077, 0.298405, 0.290784], [0.155648, 0.145744, 0.142172]),
    ]),
    "valid": transforms.Compose([
        transforms.Normalize([0.277077, 0.298405, 0.290784], [0.155648, 0.145744, 0.142172]),
    ])
}


class RSDataset(Dataset):
    """"""

    def __init__(self, split: str, data_dir:str):
        """"""
        assert split in ("train", "valid", "test")

        self.data_dir = os.path.join(data_dir, split)
        self.num = len(os.listdir(os.path.join(self.data_dir, "masks")))
        self.transform = data_transforms[split]

    def __len__(self):
        """"""
        return self.num

    def __getitem__(self, index):
        """"""
        img = torch.Tensor(np.load(os.path.join(self.data_dir, "images", f"{index}.npy")) / 255)
        img = self.transform(img)
        label = torch.Tensor(np.load(os.path.join(self.data_dir, "masks", f"{index}.npy"))).long()
        return {"inputs": img, "labels": label}


if __name__ == "__main__":

    ds = RSDataset("train", "./2021LoveDA_urban224")
