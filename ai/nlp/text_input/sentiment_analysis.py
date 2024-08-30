import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel

class SentimentAnalysisDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

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
            'label': torch.tensor(label, dtype=torch.long)
        }

class SentimentAnalysisModel(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(SentimentAnalysisModel, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.fc(pooled_output)
        return outputs

def train_sentiment_analysis_model(data, epochs, batch_size, learning_rate):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = SentimentAnalysisDataset(data, tokenizer, max_len=512)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = SentimentAnalysisModel(bert_model, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    model.eval()
    return model

# Example usage
data = pd.read_csv('sentiment_data.csv')
model = train_sentiment_analysis_model(data, epochs=5, batch_size=32, learning_rate=1e-5)

def predict_sentiment(text, model, tokenizer):
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
    _, predicted = torch.max(outputs, 1)

    return predicted.item()

# Example usage
text = "I love this product!"
sentiment = predict_sentiment(text, model, tokenizer)
print(f'Sentiment: {sentiment}')  # Output: Sentiment: 2 (Positive)
