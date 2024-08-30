import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from content_generation.utils import load_dataset, create_data_loader, train_model, evaluate_model

class ArticleGenerator(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(ArticleGenerator, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids):
        outputs = self.bert_model(input_ids)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        outputs = self.fc(sequence_output)
        return outputs

def train_article_generator(data_file, epochs, batch_size, learning_rate):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs, labels = load_dataset(data_file, tokenizer, max_len=512)
    data_loader = create_data_loader(inputs, labels, batch_size=batch_size, shuffle=True)

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = ArticleGenerator(bert_model, num_classes=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss = train_model(model, data_loader, optimizer, criterion, device='cuda')
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    return model

def generate_article(model, prompt, max_length=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer.encode_plus(
        prompt,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].flatten()
    attention_mask = encoding['attention_mask'].flatten()

    outputs = model(input_ids.unsqueeze(0))
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
model = train_article_generator('article_data.csv', epochs=5, batch_size=32, learning_rate=1e-5)
prompt = "The impact of climate change on"
article = generate_article(model, prompt)
print(article)
