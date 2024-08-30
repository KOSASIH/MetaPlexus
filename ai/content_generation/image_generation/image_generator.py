import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from image_generation.utils import load_image, create_data_loader, train_model, evaluate_model

class ImageGenerator(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(ImageGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        )

    def forward(self, image):
        encoded_image = self.encoder(image)
        decoded_image = self.decoder(encoded_image)
        return decoded_image

def train_image_generator(image_files, epochs, batch_size, learning_rate):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_loader = create_data_loader(image_files, transform, batch_size=batch_size, shuffle=True)

    model = ImageGenerator(num_layers=2, hidden_size=256)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss = train_model(model, data_loader, optimizer, criterion, device='cuda')
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    return model

def generate_image(model, noise_size):
    noise = torch.randn(1, 3, noise_size, noise_size).to('cuda')
    output = model(noise)
    return output.detach().cpu().numpy()

# Example usage
image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']
model = train_image_generator(image_files, epochs=5, batch_size=32, learning_rate=1e-4)
generated_image = generate_image(model, noise_size=256)
