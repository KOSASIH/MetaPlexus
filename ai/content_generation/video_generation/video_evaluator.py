import torch
import torch.nn as nn
from torchvision import models
from video_generation.utils import load_video, create_data_loader, evaluate_model

class VideoEvaluator(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(VideoEvaluator, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, frames):
        h0 = torch.zeros(self.lstm.num_layers, frames.size(0), self.lstm.hidden_size).to(frames.device)
        c0 = torch.zeros(self.lstm.num_layers, frames.size(0), self.lstm.hidden_size).to(frames.device)

        out, _ = self.lstm(frames, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def evaluate_video(model, video_file):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frames = load_video(video_file, transform)
    frames = torch.tensor(frames).unsqueeze(0)
    outputs = model(frames)
    score = outputs.item()
    return score

def evaluate_generated_video(model, generated_video):
    frames = torch.tensor(generated_video).unsqueeze(0)
    outputs = model(frames)
    score = outputs.item()
    return score

# Example usage
model = VideoEvaluator(num_layers=2, hidden_size=256)
video_file = 'video.mp4'
score = evaluate_video(model, video_file)
print(f'Video evaluation score: {score:.4f}')

generated_video = generate_video(model, num_frames=100, frame_size=256)
score = evaluate_generated_video(model, generated_video)
print(f'Generated video score: {score:.4f}')
