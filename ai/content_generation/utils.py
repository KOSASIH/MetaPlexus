import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel

def load_dataset(file_path, tokenizer, max_len):
    dataset = pd.read_csv(file_path)
    inputs = []
    labels = []
    for idx, row in dataset.iterrows():
        text = row['text']
        label = row['label']
        encoding = tokenizer.encode_plus(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        inputs.append(encoding['input_ids'].flatten())
        labels.append(label)
    return inputs, labels

def create_data_loader(inputs, labels, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(torch.tensor(inputs), torch.tensor(labels))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
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
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / len(data_loader.dataset)
    return accuracy
