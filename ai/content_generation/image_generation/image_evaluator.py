import torch
import torch.nn as nn
from torchvision import models
from image_generation.utils import load_image, create_data_loader, evaluate_model

class ImageEvaluator(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(ImageEvaluator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(128*4*4, 1)

    def forward(self, image):
        encoded_image = self.encoder(image)
        encoded_image = encoded_image.view(-1, 128*4*4)
        score = self.fc(encoded_image)
        return score

def evaluate_image(model, image_file):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = load_image(image_file, transform)
    image = image.unsqueeze(0)
    output = model(image)
    score = output.item()
    return score

def evaluate_generated_image(model, generated_image):
    generated_image = torch.tensor(generated_image).unsqueeze(0)
    output = model(generated_image)
    score = output.item()
    return score

# Example usage
model = ImageEvaluator(num_layers=2, hidden_size=256)
image_file = 'image.jpg'
score = evaluate_image(model, image_file)
print(f'Image evaluation score: {score:.4f}')

generated_image = generate_image(ImageGenerator(num_layers=2, hidden_size=256), noise_size=256)
score = evaluate_generated_image(model, generated_image)
print(f'Generated image score: {score:.4f}')
