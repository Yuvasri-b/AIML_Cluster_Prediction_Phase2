import pandas as pd
import joblib
import time
from sklearn.preprocessing import LabelEncoder
import subprocess

# Step 1: Load the trained model for Network Usage prediction
model = joblib.load("network_arima.pkl")  # Replace with your actual model file

# Step 2: Load sys_failure.csv
df = pd.read_csv("sys_failure.csv")

# Step 2.1: Preprocess the data (handle percentage signs and categorical encoding)
# Convert Percentage Strings to Float
for col in ["CPU_Usage", "Memory_Usage", "Disk_Usage"]:
    df[col] = df[col].str.rstrip('%').astype(float)

# Encode categorical columns (such as 'Network_Usage' and 'Pod_Status')
label_cols = ["Network_Usage", "Pod_Status", "K8s_Event_Log", "System_Log", "Network_Error"]
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store for later decoding

# Step 2.2: Convert Timestamp to datetime and set as index
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Step 2.3: Define the remediation functions

def scale_pods(deployment_name, namespace="default", replicas=3):
    command = f"kubectl scale deployment {deployment_name} --replicas={replicas} -n {namespace}"
    subprocess.run(command, shell=True)
    print(f"Scaling the deployment {deployment_name} to {replicas} replicas.")

def restart_pod(pod_name, namespace="default"):
    command = f"kubectl delete pod {pod_name} -n {namespace}"
    subprocess.run(command, shell=True)
    print(f"Restarting pod {pod_name}...")

def enable_autoscaling(deployment_name, min_replicas=1, max_replicas=10, cpu_utilization=50):
    command = f"kubectl autoscale deployment {deployment_name} --cpu-percent={cpu_utilization} --min={min_replicas} --max={max_replicas}"
    subprocess.run(command, shell=True)
    print(f"Enabled autoscaling on {deployment_name} with min={min_replicas}, max={max_replicas} replicas.")

# Step 2.4: Real-time prediction (simulated) loop for Network Usage anomalies
print("\nStarting Real-Time Monitoring...\n")

# Use a sliding window approach to keep track of previous data points
window_size = 10  # Adjust the number of previous points used for prediction

# Iterate over the rows in the dataframe by integer index (using `df.index` for time-series access)
for i in range(window_size, len(df)):
    # Extract the last `window_size` data points for prediction
    input_data = df['Network_Usage'].iloc[i - window_size:i]
    prediction = model.forecast(steps=1)  # Now it's an array, no need to index [0]

    # Access the forecasted value using .iloc for position-based access
    try:
        predicted_value = prediction.iloc[0]  # Access the first value in the prediction result (using .iloc)
    except Exception as e:
        print(f"Error accessing prediction value: {e}")
        continue  # Skip to next iteration if there's an error

    node_id = df['Node'].iloc[i]
    timestamp = df.index[i]  # The index (timestamp) of the current row
    net_usage = df['Network_Usage'].iloc[i]

    # Take action if network usage exceeds the threshold or ARIMA predicts high usage
    if predicted_value > 80 or net_usage > 80:
        print(f"[{timestamp}]  High Network Usage detected on Node {node_id} ({net_usage}%) -> Taking action...")
        
        
    else:
        print(f"[{timestamp}]  Node {node_id} healthy, Network usage: {net_usage}%")
    
    time.sleep(1)  # Simulate delay for real-time monitoring
