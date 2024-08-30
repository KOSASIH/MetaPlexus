**iot_device_connectivity.py**
```python
import socket
import json
import time
import threading
import paho.mqtt.client as mqtt
from cryptography.fernet import Fernet

# Load IoT device configuration
with open("iot_device_config.json", "r") as f:
    config = json.load(f)

# Define IoT device connectivity class
class IoTDeviceConnectivity:
    def __init__(self, device_id, device_type, mqtt_broker, mqtt_port):
        self.device_id = device_id
        self.device_type = device_type
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.fernet = Fernet(config["encryption_key"])

    def on_connect(self, client, userdata, flags, rc):
        print("Connected to MQTT broker with result code " + str(rc))
        self.mqtt_client.subscribe("iot/devices/" + self.device_id + "/commands")

    def on_message(self, client, userdata, msg):
        print("Received message on topic " + msg.topic + ": " + msg.payload.decode())
        if msg.topic == "iot/devices/" + self.device_id + "/commands":
            self.process_command(msg.payload.decode())

    def process_command(self, command):
        if command == "start_data_stream":
            self.start_data_stream()
        elif command == "stop_data_stream":
            self.stop_data_stream()
        else:
            print("Unknown command: " + command)

    def start_data_stream(self):
        print("Starting data stream...")
        self.socket.connect((self.mqtt_broker, self.mqtt_port))
        self.socket.sendall(self.fernet.encrypt(b"start_data_stream"))
        self.data_stream_thread = threading.Thread(target=self.send_data_stream)
        self.data_stream_thread.start()

    def stop_data_stream(self):
        print("Stopping data stream...")
        self.socket.sendall(self.fernet.encrypt(b"stop_data_stream"))
        self.data_stream_thread.join()

    def send_data_stream(self):
        while True:
            data = self.read_sensor_data()
            self.socket.sendall(self.fernet.encrypt(data.encode()))
            time.sleep(1)

    def read_sensor_data(self):
        # Replace with actual sensor data reading code
        return "Sensor data: " + str(time.time())

    def connect_to_mqtt(self):
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port)
        self.mqtt_client.loop_start()

    def disconnect_from_mqtt(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

# Create IoT device connectivity instance
iot_device = IoTDeviceConnectivity("device_1", "temperature_sensor", "localhost", 1883)

# Connect to MQTT broker
iot_device.connect_to_mqtt()

# Start data stream
iot_device.start_data_stream()

# Run indefinitely
while True:
    time.sleep(1)
