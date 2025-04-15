import pandas as pd
import joblib
import time
from sklearn.preprocessing import LabelEncoder
from kubernetes import client, config

# Step 1: Load the trained Isolation Forest model for service disruption prediction
iso_forest = joblib.load("service_iso_forest.pkl")  # Replace with your model file name

# Step 2: Load and preprocess the dataset (system failure logs)
df = pd.read_csv("sys_failure.csv")

# Clean percentage columns (convert to float)
for col in ["CPU_Usage", "Memory_Usage", "Disk_Usage"]:
    df[col] = df[col].str.rstrip('%').astype(float)

# Encode categorical columns like 'Network_Usage', 'Pod_Status', 'K8s_Event_Log', etc.
label_cols = ["Network_Usage", "Pod_Status", "K8s_Event_Log", "System_Log", "Network_Error"]
encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encode categorical values
    encoders[col] = le  # Store for later decoding

# Define feature columns used by the model
feature_cols = ["CPU_Usage", "Memory_Usage", "Disk_Usage", "Network_Usage", "Pod_Status", "K8s_Event_Log", "System_Log", "Network_Error"]

# Step 3: Load kubeconfig and initialize Kubernetes API client
config.load_kube_config()  # This loads the kubeconfig file (same file used by kubectl)

# Step 4: Function to restart the failed pods
def restart_failed_pods(node_name):
    v1 = client.CoreV1Api()  # Initialize the Kubernetes API client
    pods = v1.list_pod_for_all_namespaces(field_selector=f'spec.nodeName={node_name}').items  # Get all pods on this node
    
    for pod in pods:
        if pod.status.phase == "Running" and pod.metadata.namespace != "kube-system":
            print(f"Restarting Pod: {pod.metadata.name}")
            v1.delete_namespaced_pod(name=pod.metadata.name, namespace=pod.metadata.namespace)  # Delete (restart) the pod

# Step 5: Function to recommend actions based on service disruption
def recommend_actions(node_name):
    print(f"Recommendation: Service disruption detected on Node {node_name}.")
    print("Suggested actions:")
    print("1. Restart the pod to reallocate resources.")
    print("2. Check the system logs for any critical errors.")
    print("3. Ensure that network configurations are optimized.")
    print("4. Monitor CPU/Memory usage for abnormal spikes.")

# Step 6: Real-time prediction loop with automated and recommended actions
print("\nStarting Service Disruption Monitoring...\n")

# Iterate over each row in the dataframe
for i, row in df.iterrows():
    input_data = pd.DataFrame([row[feature_cols]], columns=feature_cols)  # Reshape row to DataFrame for model input
    anomaly_score = iso_forest.predict(input_data)[0]  # Predict if anomaly (service disruption) is detected
    
    if anomaly_score == -1:  # If anomaly is detected (-1 = disruption)
        print(f"[{row['Timestamp']}] Service Disruption predicted on Node: {row['Node']}")
        
        # Trigger automated remediation (restart pods)
        restart_failed_pods(row['Node'])
        
        # In addition to automation, recommend additional actions
        recommend_actions(row['Node'])
    
    else:
        print(f"[{row['Timestamp']}] Node {row['Node']} healthy")
    
    time.sleep(0.2)  # Simulate a real-time delay (this can be adjusted as needed)
