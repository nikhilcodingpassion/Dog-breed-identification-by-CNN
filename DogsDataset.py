import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np


class DogsDataset(Dataset):
    def __init__(self, csv_file='', root_dir="/", transform=None, mode='train'):
        self.mode = mode
        self.labels = self.parseData(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def parseData(self, file_path):
        data = pd.read_csv(file_path)
        if self.mode == 'train':
            # Assuming the training data still has a 'breed' column
            breed_to_label = {breed: idx for idx, breed in enumerate(data['breed'].unique())}
            data['label'] = data['breed'].map(breed_to_label)
        else:
            # For test data, we don't have a 'breed' column
            # Just use the 'id' column
            data['label'] = 0  # Dummy label for test data
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)

        if self.mode == 'train':
            label = self.labels.iloc[idx]['label']
        else:
            # Use the filename (or a part of it) as the 'label' for test images
            label = self.labels.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        return image, label