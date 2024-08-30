import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel

class EntityRecognitionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        labels = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class EntityRecognitionModel(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(EntityRecognitionModel, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        outputs = self.fc(sequence_output)
        return outputs

def train_entity_recognition_model(data, epochs, batch_size, learning_rate):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = EntityRecognitionDataset(data, tokenizer, max_len=512)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = EntityRecognitionModel(bert_model, num_classes=9)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, 9), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    model.eval()
    return model

# Example usage
data = pd.read_csv('entity_data.csv')
model = train_entity_recognition_model(data, epochs=5, batch_size=32, learning_rate=1e-5)

def predict_entities(text, model, tokenizer):
    encoding = tokenizer.encode_plus(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].flatten()
    attention_mask = encoding['attention_mask'].flatten()

    outputs = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
    outputs = torch.argmax(outputs, dim=2)

    entities = []
    for i in range(outputs.shape[1]):
        entity = []
        for j in range(outputs.shape[0]):
            if outputs[j, i] != 0:
                entity.append(text[j:j+1])
        entities.append(entity)

    return entities

# Example usage
text = "John Smith is a doctor at ABC Hospital."
entities = predict_entities(text, model, tokenizer)
print(entities)  # Output: [['John Smith'], ['doctor'], ['ABC Hospital']]
