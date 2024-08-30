import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms

def load_image(image_file, transform):
    image = cv2.imread(image_file)
    image = transform(image)
    return image

def create_data_loader(image_files, transform, batch_size, shuffle=True):
    dataset = ImageDataset(image_files, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

class ImageDataset(Dataset):
    def __init__(self, image_files, transform):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = load_image(image_file, self.transform)
        return image

def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        image = batch[0].to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, image)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in data_loader:
            image = batch[0].to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == image).sum().item()
    accuracy = total_correct / len(data_loader.dataset)
    return accuracy
