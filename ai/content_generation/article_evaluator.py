import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from content_generation.utils import load_dataset, create_data_loader, evaluate_model

class ArticleEvaluator(nn.Module):
    def __init__(self, bert_model):
        super(ArticleEvaluator, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids):
        outputs = self.bert_model(input_ids)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        outputs = self.fc(sequence_output)
        return outputs

def evaluate_article(model, data_file, batch_size):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs, labels = load_dataset(data_file, tokenizer, max_len=512)
    data_loader = create_data_loader(inputs, labels, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    device = 'cuda'
    model.to(device)

    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate_generated_article(model, generated_article):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer.encode_plus(
        generated_article,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].flatten()
    attention_mask = encoding['attention_mask'].flatten()

    outputs = model(input_ids.unsqueeze(0))
    score = outputs.item()
    return score

# Example usage
model = ArticleEvaluator(BertModel.from_pretrained('bert-base-uncased'))
data_file = 'article_eval_data.csv'
batch_size = 32
score = evaluate_article(model, data_file, batch_size)
print(f'Article evaluation score: {score:.4f}')

generated_article = "This is a sample generated article."
score = evaluate_generated_article(model, generated_article)
print(f'Generated article score: {score:.4f}')
