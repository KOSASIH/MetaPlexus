**mr_experience.py**
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
import pyopenvr
import azure.spatial_anchors as ASA

# Load MR experience configuration
with open("mr_experience_config.json", "r") as f:
    config = json.load(f)

# Define MR experience class
class MRExperience:
    def __init__(self, model_path, label_encoder_path, scene_graph_path, hand_landmark_model_path, vr_system, ar_cloud_anchor):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.scene_graph_path = scene_graph_path
        self.hand_landmark_model_path = hand_landmark_model_path
        self.vr_system = vr_system
        self.ar_cloud_anchor = ar_cloud_anchor
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

    def create_cloud_anchor(self, image, object_label):
        # Create cloud anchor using Azure Spatial Anchors
        cloud_anchor = ASA.CloudSpatialAnchor()
        cloud_anchor.create_anchor(image, object_label)
        return cloud_anchor

    def interact_with_object(self, image, hand_landmarks, cloud_anchor):
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

    def render_mr_scene(self, scene_graph_node, cloud_anchor):
        # Render MR scene using OpenVR and Azure Spatial Anchors
        mr_scene = pyopenvr.render_scene(scene_graph_node, cloud_anchor)
        return mr_scene

    def start_mr_experience(self):
        # Initialize VR system
        self.vr_system.init()

        # Start MR experience loop
        while True:
            # Get VR tracking data
            tracking_data = self.vr_system.get_tracking_data()

            # Create cloud anchor
            cloud_anchor = self.create_cloud_anchor(tracking_data["image"], tracking_data["object_label"])
                        # Estimate hand landmarks
            hand_landmarks = self.estimate_hand_landmarks(tracking_data["image"])

            # Interact with object
            scene_graph_node = self.interact_with_object(tracking_data["image"], hand_landmarks, cloud_anchor)

            # Render MR scene
            mr_scene = self.render_mr_scene(scene_graph_node, cloud_anchor)

            # Display MR scene
            self.vr_system.display_scene(mr_scene)

# Create MR experience instance
mr_experience = MRExperience("model.pth", "label_encoder.json", "scene_graph.json", "hand_landmark_model.tflite", pyopenvr.VRSystem(), ASA.CloudSpatialAnchor())

# Start MR experience
mr_experience.start_mr_experience()
