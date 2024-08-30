import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import os
import pandas as pd

class CustomDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file)
        if self.transform:
            image = self.transform(image)
        return image

def create_data_loader(data_dir, batch_size, transform=None, shuffle=True):
    dataset = CustomDataset(data_dir, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def create_data_loader_from_csv(csv_file, data_dir, batch_size, transform=None, shuffle=True):
    df = pd.read_csv(csv_file)
    image_files = [os.path.join(data_dir, f) for f in df['image_file']]
    labels = df['label']
    dataset = CustomDataset(image_files, labels, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

class CustomDatasetWithLabels(data.Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label = self.labels[idx]
        image = Image.open(image_file)
        if self.transform:
            image = self.transform(image)
        return image, label
