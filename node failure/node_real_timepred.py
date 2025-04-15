import pandas as pd
import joblib
import time
from sklearn.preprocessing import LabelEncoder
from kubernetes import client, config

# Step 1: Load the trained Node/Pod failure model
model = joblib.load("pod_random_forest.pkl")  # or replace with clean_node_xgb_model.pkl

# Step 2: Load and preprocess dataset
df = pd.read_csv("sys_failure.csv")

# Clean percentage columns
for col in ["CPU_Usage", "Memory_Usage", "Disk_Usage"]:
    df[col] = df[col].astype(str).str.rstrip('%').astype(float)

# Encode 'Network_Usage' (same as in training)
le = LabelEncoder()
df["Network_Usage"] = le.fit_transform(df["Network_Usage"])

# Define feature columns
feature_cols = ["CPU_Usage", "Memory_Usage", "Disk_Usage", "Network_Usage"]

print("\nStarting Node/Pod Failure Monitoring...\n")

# Step 3: Load kubeconfig and initialize Kubernetes API client
config.load_kube_config()  # This loads the kubeconfig file (same file used by kubectl)
v1 = client.CoreV1Api()

# Step 4: Function to restart failed pods
def restart_failed_pods(node_name):
    v1 = client.CoreV1Api()  # Initialize the Kubernetes API client
    try:
        pods = v1.list_pod_for_all_namespaces(
            field_selector=f'spec.nodeName={node_name}',
            timeout_seconds=60  # Increase the timeout to 60 seconds
        ).items
    except client.exceptions.ApiException as e:
        print(f"Error fetching pods for node {node_name}: {e}")
        return

    for pod in pods:
        if pod.status.phase == "Running" and pod.metadata.namespace != "kube-system":
            print(f"Deactivating Pod: {pod.metadata.name}")
            v1.delete_namespaced_pod(name=pod.metadata.name, namespace=pod.metadata.namespace)  # Delete (restart) the pod


# Step 5: Real-time prediction loop
for i, row in df.iterrows():
    input_data = pd.DataFrame([row[feature_cols]], columns=feature_cols)  # Reshape row to DataFrame for model input
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        print(f"[{row['Timestamp']}] Pod Failure predicted on Node: {row['Node']}")
        
        # Trigger automated remediation: Restart the pod
        restart_failed_pods(row['Node'])
    else:
        print(f"[{row['Timestamp']}] Node {row['Node']} healthy")

    time.sleep(0.2)  # Simulate a real-time delay (this can be adjusted as needed)
