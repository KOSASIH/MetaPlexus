**ar_object_recognition.py**
```python
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import DenseNet
from sklearn.preprocessing import LabelEncoder
import json
import os

# Load AR object recognition configuration
with open("ar_object_recognition_config.json", "r") as f:
    config = json.load(f)

# Define AR object recognition class
class ARObjectRecognition:
    def __init__(self, model_path, label_encoder_path):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.label_encoder = self.load_label_encoder()

    def load_model(self):
        model = DenseNet(num_classes=config["num_classes"])
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        return model

    def load_label_encoder(self):
        with open(self.label_encoder_path, "r") as f:
            label_encoder = LabelEncoder()
            label_encoder.classes_ = json.load(f)
        return label_encoder

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def recognize_object(self, image):
        image = self.preprocess_image(image)
        image = image.to(self.device)
        output = self.model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        label = self.label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        return label

    def start_video_capture(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("AR Object Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            object_label = self.recognize_object(frame)
            print("Recognized object: " + object_label)
        cap.release()
        cv2.destroyAllWindows()

# Create AR object recognition instance
ar_object_recognition = ARObjectRecognition("model.pth", "label_encoder.json")

# Start video capture
ar_object_recognition.start_video_capture()
