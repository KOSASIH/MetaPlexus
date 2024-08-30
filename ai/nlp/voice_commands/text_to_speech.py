import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io.wavfile import write

class TextToSpeechDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        mel_spectrogram = self.data.iloc[idx, 1]

        # Preprocess text
        text = text.lower()
        text = torch.tensor([ord(c) for c in text], dtype=torch.long)

        # Preprocess mel spectrogram
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

        return {
            'text': text,
            'mel_spectrogram': mel_spectrogram
        }

class TextToSpeechModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextToSpeechModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, output_dim, num_layers=2, batch_first=True)

    def forward(self, text):
        h0 = torch.zeros(2, text.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2, text.size(0), self.hidden_dim).to(device)

        out, _ = self.encoder(text, (h0, c0))
        out, _ = self.decoder(out, (h0, c0))
        return out

def train_text_to_speech_model(data, epochs, batch_size, learning_rate):
    max_len = 100
    dataset = TextToSpeechDataset(data, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = 128
    hidden_dim = 256
    output_dim = 128
    model = TextToSpeechModel(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            text = batch['text'].to(device)
            mel_spectrogram = batch['mel_spectrogram'].to(device)

            optimizer.zero_grad()

            outputs = model(text)
            loss = criterion(outputs, mel_spectrogram)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    model.eval()
    return model

# Example usage
data = pd.read_csv('text_data.csv')
model = train_text_to_speech_model(data, epochs=5, batch_size=32, learning_rate=1e-5)

# Synthesize audio
def synthesize_audio(text, model):
    text = torch.tensor([ord(c) for c in text], dtype=torch.long)
    text = text.unsqueeze(0)

    mel_spectrogram = model(text)
    mel_spectrogram = mel_spectrogram.squeeze(0)

    # Convert mel spectrogram to audio
    audio = mel_spectrogram.detach().numpy()
    audio = audio.T
    audio = librosa.feature.inverse_mel_spectrogram(audio, sr=22050, n_fft=1024, hop_length=256)
    audio = audio.astype(np.int16)

    return audio

# Example usage
text = "Hello, world!"
audio = synthesize_audio(text, model)
write("output.wav", 22050, audio)
