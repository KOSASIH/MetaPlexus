**ar_scene_understanding.py**
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
import open3d as o3d

# Load AR scene understanding configuration
with open("ar_scene_understanding_config.json", "r") as f:
    config = json.load(f)

# Define AR scene understanding class
class ARSceneUnderstanding:
    def __init__(self, model_path, label_encoder_path, scene_graph_path):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.scene_graph_path = scene_graph_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.label_encoder = self.load_label_encoder()
        self.scene_graph = self.load_scene_graph()

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

    def load_scene_graph(self):
        with open(self.scene_graph_path, "r") as f:
            scene_graph = json.load(f)
        return scene_graph

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

    def estimate_depth(self, image):
        # Use a depth estimation model (e.g. MiDaS) to estimate depth
        # ...
        return depth_map

    def reconstruct_scene(self, image, depth_map):
        # Use a 3D reconstruction algorithm (e.g. COLMAP) to reconstruct the scene
        # ...
        return point_cloud

    def understand_scene(self, image):
        object_label = self.recognize_object(image)
        depth_map = self.estimate_depth(image)
        point_cloud = self.reconstruct_scene(image, depth_map)
        scene_graph_node = self.scene_graph[object_label]
        return scene_graph_node, point_cloud

    def start_video_capture(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("AR Scene Understanding", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            scene_graph_node, point_cloud = self.understand_scene(frame)
            print("Recognized object: " + scene_graph_node["label"])
            print("Scene graph node: " + str(scene_graph_node))
            o3d.visualization.draw_geometries([point_cloud])
        cap.release()
        cv2.destroyAllWindows()

# Create AR scene understanding instance
ar_scene_understanding = ARSceneUnderstanding("model.pth", "label_encoder.json", "scene_graph.json")

# Start video capture
ar_scene_understanding.start_video_capture()
