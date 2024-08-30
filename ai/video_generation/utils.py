import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms

def load_video(video_file, transform):
    video = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = transform(frame)
        frames.append(frame)
    video.release()
    return frames

def create_data_loader(video_files, transform, batch_size, shuffle=True):
    dataset = VideoDataset(video_files, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

class VideoDataset(Dataset):
    def __init__(self, video_files, transform):
        self.video_files = video_files
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        frames = load_video(video_file, self.transform)
        return frames

def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        frames = batch[0].to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, frames)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in data_loader:
            frames = batch[0].to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == frames).sum().item()
    accuracy = total_correct / len(data_loader.dataset)
    return accuracy
