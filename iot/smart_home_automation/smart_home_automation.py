**smart_home_automation.py**
```python
import socket
import json
import time
import threading
import paho.mqtt.client as mqtt
from cryptography.fernet import Fernet
import RPi.GPIO as GPIO

# Load smart home automation configuration
with open("smart_home_config.json", "r") as f:
    config = json.load(f)

# Define smart home automation class
class SmartHomeAutomation:
    def __init__(self, mqtt_broker, mqtt_port):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.fernet = Fernet(config["encryption_key"])
        self.GPIO_setup()

    def GPIO_setup(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT)  # Living room light
        GPIO.setup(23, GPIO.OUT)  # Kitchen light
        GPIO.setup(24, GPIO.OUT)  # Bedroom light

    def on_connect(self, client, userdata, flags, rc):
        print("Connected to MQTT broker with result code " + str(rc))
        self.mqtt_client.subscribe("smart_home/commands")

    def on_message(self, client, userdata, msg):
        print("Received message on topic " + msg.topic + ": " + msg.payload.decode())
        if msg.topic == "smart_home/commands":
            self.process_command(msg.payload.decode())

    def process_command(self, command):
        if command == "turn_on_living_room_light":
            self.turn_on_light(17)
        elif command == "turn_off_living_room_light":
            self.turn_off_light(17)
        elif command == "turn_on_kitchen_light":
            self.turn_on_light(23)
        elif command == "turn_off_kitchen_light":
            self.turn_off_light(23)
        elif command == "turn_on_bedroom_light":
            self.turn_on_light(24)
        elif command == "turn_off_bedroom_light":
            self.turn_off_light(24)
        else:
            print("Unknown command: " + command)

    def turn_on_light(self, pin):
        GPIO.output(pin, GPIO.HIGH)

    def turn_off_light(self, pin):
        GPIO.output(pin, GPIO.LOW)

    def connect_to_mqtt(self):
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port)
        self.mqtt_client.loop_start()

    def disconnect_from_mqtt(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

# Create smart home automation instance
smart_home = SmartHomeAutomation("localhost", 1883)

# Connect to MQTT broker
smart_home.connect_to_mqtt()

# Run indefinitely
while True:
    time.sleep(1)
