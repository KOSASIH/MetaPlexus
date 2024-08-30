**ar_interaction.py**
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
import mediapipe as mp

# Load AR interaction configuration
with open("ar_interaction_config.json", "r") as f:
    config = json.load(f)

# Define AR interaction class
class ARInteraction:
    def __init__(self, model_path, label_encoder_path, scene_graph_path, hand_landmark_model_path):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.scene_graph_path = scene_graph_path
        self.hand_landmark_model_path = hand_landmark_model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.label_encoder = self.load_label_encoder()
        self.scene_graph = self.load_scene_graph()
        self.hand_landmark_model = self.load_hand_landmark_model()

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

    def load_hand_landmark_model(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        return hands

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

    def estimate_hand_landmarks(self, image):
        results = self.hand_landmark_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        else:
            return None

    def interact_with_object(self, image, hand_landmarks):
        object_label = self.recognize_object(image)
        scene_graph_node = self.scene_graph[object_label]
        if hand_landmarks:
            finger_tips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                           hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                           hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                           hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
            distances = [np.linalg.norm(np.array(finger_tip) - np.array(scene_graph_node["position"])) for finger_tip in finger_tips]
            if min(distances) < config["interaction_distance"]:
                print("Interaction detected!")
                # Perform interaction action (e.g. pick up object, rotate object, etc.)
                # ...
        return scene_graph_node

    def start_video_capture(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("AR Interaction", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            hand_landmarks = self.estimate_hand_landmarks(frame)
            scene_graph_node = self.interact_with_object(frame, hand_landmarks)
            print("Scene graph node: " + str(scene_graph_node))
        cap.release()
        cv2.destroyAllWindows()

# Create AR interaction instance
ar_interaction = ARInteraction("model.pth", "label_encoder.json", "scene_graph.json", "hand_landmark_model.tflite")

# Start video capture
ar_interaction.start_video_capture()
