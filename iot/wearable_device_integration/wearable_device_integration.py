**wearable_device_integration.py**
```python
import bluetooth
import json
import time
import threading
from scipy.signal import filter_design
from scipy.signal import lfilter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load wearable device configuration
with open("wearable_device_config.json", "r") as f:
    config = json.load(f)

# Define wearable device integration class
class WearableDeviceIntegration:
    def __init__(self, device_id, device_type, bluetooth_address):
        self.device_id = device_id
        self.device_type = device_type
        self.bluetooth_address = bluetooth_address
        self.bluetooth_socket = bluetooth.socket(bluetooth.RFCOMM)
        self.data_buffer = []
        self.data_thread = threading.Thread(target=self.read_data_from_device)
        self.data_thread.daemon = True
        self.data_thread.start()
        self.model = self.train_machine_learning_model()

    def connect_to_device(self):
        print("Connecting to wearable device...")
        self.bluetooth_socket.connect((self.bluetooth_address, 1))
        print("Connected to wearable device.")

    def read_data_from_device(self):
        while True:
            data = self.bluetooth_socket.recv(1024)
            self.data_buffer.append(data.decode())
            time.sleep(0.1)

    def process_data(self):
        while True:
            if len(self.data_buffer) > 0:
                data = self.data_buffer.pop(0)
                self.analyze_data(data)
            time.sleep(0.1)

    def analyze_data(self, data):
        # Filter data using a Butterworth filter
        nyq = 0.5 * 100  # Nyquist frequency
        cutoff = 10  # Cutoff frequency
        order = 4  # Filter order
        b, a = filter_design.butter(order, cutoff / nyq, btype='low')
        filtered_data = lfilter(b, a, data)
        # Extract features from filtered data
        features = self.extract_features(filtered_data)
        # Classify data using machine learning model
        classification = self.model.predict(features)
        print("Classification: " + str(classification))

    def extract_features(self, data):
        # Extract mean, standard deviation, and peak values from data
        mean = np.mean(data)
        std = np.std(data)
        peak = np.max(data)
        return [mean, std, peak]

    def train_machine_learning_model(self):
        # Load training data
        data = pd.read_csv("training_data.csv")
        X = data.drop("label", axis=1)
        y = data["label"]
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train machine learning model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Model accuracy: " + str(accuracy))
        return model

    def start_data_processing(self):
        self.process_thread = threading.Thread(target=self.process_data)
        self.process_thread.daemon = True
        self.process_thread.start()

# Create wearable device integration instance
wearable_device = WearableDeviceIntegration("device_1", "fitness_tracker", "00:11:22:33:44:55")

# Connect to wearable device
wearable_device.connect_to_device()

# Start data processing
wearable_device.start_data_processing()

# Run indefinitely
while True:
    time.sleep(1)
