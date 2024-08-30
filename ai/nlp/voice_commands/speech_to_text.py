import speech_recognition as sr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SpeechToTextDataset(Dataset):
    def __init__(self, data, sample_rate, window_size):
        self.data = data
        self.sample_rate = sample_rate
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, transcript = self.data.iloc[idx, 0], self.data.iloc[idx, 1]

        # Preprocess audio
        audio = sr.AudioData(audio, sample_rate=self.sample_rate)
        audio = audio.get_array_of_samples()
        audio = torch.tensor(audio, dtype=torch.float32)

        # Preprocess transcript
        transcript = transcript.lower()
        transcript = torch.tensor([ord(c) for c in transcript], dtype=torch.long)

        return {
            'audio': audio,
            'transcript': transcript
        }

class SpeechToTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechToTextModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3)
        self.fc1 = nn.Linear(hidden_dim * self.window_size, output_dim)

    def forward(self, audio):
        x = torch.relu(self.conv1(audio))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, self.window_size * hidden_dim)
        x = self.fc1(x)
        return x

def train_speech_to_text_model(data, epochs, batch_size, learning_rate):
    sample_rate = 22050
    window_size = 1024
    dataset = SpeechToTextDataset(data, sample_rate, window_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = 1
    hidden_dim = 128
    output_dim = 128
    model = SpeechToTextModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            audio = batch['audio'].to(device)
            transcript = batch['transcript'].to(device)

            optimizer.zero_grad()

            outputs = model(audio)
            loss = criterion(outputs, transcript)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    model.eval()
    return model

# Example usage
data = pd.read_csv('speech_data.csv')
model = train_speech_to_text_model(data, epochs=5, batch_size=32, learning_rate=1e-5)
