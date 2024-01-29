import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

```
data organization
source/img
source/mask
train/images
train/masks
test/images
test/masks

```

class CVCDataset(Dataset):
    def __init__(self, args, data_path, mask_path, transform=None, mode='Training', plane=False):
        self.img_files = os.listdir(data_path)
        self.mask_files = os.listdir(mask_path)
        self.data_path = data_path
        self.mask_path = mask_path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        """Get the images"""
        img_name = self.img_files[index]
        mask_name = self.mask_files[index]

        img_path = os.path.join(self.data_path, img_name)
        mask_path = os.path.join(self.mask_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return (img, mask, img_name)
