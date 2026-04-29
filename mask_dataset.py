"""
@author: Lochana Marasinghe
@date: 4/28/2026
@description: 
"""
import os
import glob

import cv2
import numpy as np
from torch.utils.data import Dataset


class WheatBinaryDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_fps = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        self.mask_fps = [fp.replace(".jpg", "_mask.png") for fp in self.image_fps]
        self.transform = transform

        # PRE-LOAD INTO RAM
        print("Pre-loading images into RAM for speed...")
        self.images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in self.image_fps]
        self.masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in self.mask_fps]

    def __getitem__(self, i):
        image = self.images[i]
        mask = self.masks[i]
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask.unsqueeze(0)

    def __len__(self):
        return len(self.image_fps)