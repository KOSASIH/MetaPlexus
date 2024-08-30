import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, data_loader, optimizer, criterion, device, epochs):
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for batch in data_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / len(data_loader.dataset)
    return accuracy

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    return model
