import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import defaultdict



class DiffraNetDataset(Dataset):
    def __init__(self, data_dir, transform=None, data_type="train"):
        path2data = os.path.join(data_dir, data_type)
        self.filenames = []
        self._size = 0
        root, dirs, _ = next(os.walk(path2data))
        for class_ in dirs:
            path = os.path.join(root, class_)
            _, _, images = next(os.walk(path))
            for img in images:
                self.filenames.append((os.path.join(path ,img), class_))
        
        self.transform = transform

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path, label = self.filenames[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, label
    

    

