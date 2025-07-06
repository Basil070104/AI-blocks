import random
import time

# Simulate a temperature sensor
def get_temperature():
    return round(random.uniform(20.0, 40.0), 2)  # °C

# Edge device processing
def edge_device(threshold=35.0):
    for _ in range(10):  # Simulate 10 readings
        temp = get_temperature()
        print(f"[Edge Device] Temperature reading: {temp} °C")

        # Edge logic: Only send to cloud if above threshold
        if temp > threshold:
            send_to_cloud(temp)
        else:
            print("[Edge Device] Temperature normal. No need to send to cloud.\n")
        time.sleep(1)

# Simulated cloud processing
def send_to_cloud(data):
    print(f"[Cloud] ALERT: High temperature detected! ({data} °C)\n")

# Run the edge device
edge_device()
