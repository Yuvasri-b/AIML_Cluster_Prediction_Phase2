import pandas as pd
import joblib
from datetime import datetime
import time
import random

# Load your trained model
model = joblib.load("cpu_isolation_forest.pkl")  # Replace with actual file path

# Define feature columns used in training
feature_cols = ["CPU_Usage", "Memory_Usage", "Disk_Usage"]

def simulate_metrics():
    return {
        "CPU_Usage": round(random.uniform(30, 100), 2),
        "Memory_Usage": round(random.uniform(30, 100), 2),
        "Disk_Usage": round(random.uniform(20, 100), 2)
    }

def trigger_auto_scale(node_id):
    """Simulate an auto-scale action"""
    print(f"Triggering auto-scale for {node_id}...")

def send_alert(node_id):
    """Simulate sending an alert to the admin"""
    print(f"ALERT: Resource exhaustion predicted on {node_id}. Immediate action required!")

def reboot_node(node_id):
    """Simulate rebooting the node to resolve issues"""
    print(f"Rebooting {node_id} to recover from resource exhaustion...")

def predict_resource_exhaustion(node_id, input_data):
    """Run prediction and trigger remediation actions"""
    input_df = pd.DataFrame([input_data], columns=feature_cols)
    pred = model.predict(input_df)[0]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if pred == 1:
        # If resource exhaustion predicted, initiate remediation actions
        print(f"[{timestamp}] Resource Exhaustion predicted on Node: {node_id}")
        
        # Remediation Actions (choose one or combine multiple)
        send_alert(node_id)
        trigger_auto_scale(node_id)  # Auto-scale
        reboot_node(node_id)  # Reboot node as a last resort
        
    else:
        print(f"[{timestamp}] Node {node_id} operating normally")

# Simulate streaming metrics from multiple nodes
node_index = 0
while True:
    node = f"node-{node_index % 100}"  # Rotate through node-0 to node-99
    metrics = simulate_metrics()
    predict_resource_exhaustion(node, metrics)
    time.sleep(2)  # Wait 2 seconds before next prediction
    node_index += 1
