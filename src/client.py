import socket
import json
import pandas as pd
import joblib
import os
import csv
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from together import Together
from together.error import AuthenticationError

HOST = 'localhost'
PORT = 9999

# Load the trained model and preprocessor
model = joblib.load("src/anomaly_model.joblib")
preprocessor = joblib.load("src/preprocessor.joblib")

# Load Together API key
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise AuthenticationError(
        "Missing TOGETHER_API_KEY. Set it with:\n"
        "  - CMD: set TOGETHER_API_KEY=your_key\n"
        "  - PowerShell: $env:TOGETHER_API_KEY = 'your_key'"
    )

# Initialize Together client
client = Together(api_key=api_key)

# Initialize CSV if not present
if not os.path.exists('anomalies.csv'):
    with open('anomalies.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Data', 'Label', 'Reason', 'Score'])

def pre_process_data(data):
    df = pd.DataFrame([data])
    # Apply preprocessing pipeline
    X_transformed = preprocessor.transform(df)
    return X_transformed

def create_prompt(data):
    return [
        {
            "role": "system",
            "content": "You are an expert assistant for detecting and labeling network traffic anomalies."
        },
        {
            "role": "user",
            "content": f"""Given the following network traffic data:
{data}

Identify the anomaly type and explain the likely cause.

Respond ONLY in the following exact format, without any additional text:

Label: <short label here>
Reason: <brief explanation here>"""
        }
    ]

def log_anomaly(data, label, reason, score):
    with open('anomalies.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now(), str(data), label, reason, f"{score:.4f}"])

def plot_pca(data_vectors, labels):
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(data_vectors)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels,
                    palette={0: 'blue', 1: 'red'}, alpha=0.7)
    plt.title("PCA by Prediction Labels (Normal=blue, Anomaly=red)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"pca_plot_{timestamp}.png")
    plt.close()

# Buffers for PCA batching
X_buffer = []
label_buffer = []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = ""
    print("Client connected to server.\n")

    while True:
        chunk = s.recv(1024).decode()
        if not chunk:
            break
        buffer += chunk

        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            try:
                data = json.loads(line)
                print(f'Data Received:\n{data}\n')

                # Preprocess and predict
                X = pre_process_data(data)
                prediction = model.predict(X)
                score = model.decision_function(X)[0]
                is_anomaly = 1 if prediction[0] == -1 else 0

                print(f"Model prediction: {prediction[0]} | Is anomaly: {is_anomaly} | Score: {score:.4f}")

                # Append for PCA plot
                X_buffer.append(X[0])
                label_buffer.append(is_anomaly)

                if is_anomaly == 1:
                    messages = create_prompt(data)
                    try:
                        response = client.chat.completions.create(
                            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                            messages=messages,
                            stream=False
                        )
                        result = response.choices[0].message.content

                        # Extract label and reason
                        label = "Unknown"
                        reason = "Unknown"
                        for line in result.split('\n'):
                            if line.startswith("Label:"):
                                label = line[len("Label:"):].strip()
                            elif line.startswith("Reason:"):
                                reason = line[len("Reason:"):].strip()

                        print(f"\n🚨 Anomaly Detected!\nLabel: {label}\nReason: {reason}\n")
                        log_anomaly(data, label, reason, score)

                    except Exception as e:
                        print(f"Error with Together AI API: {e}")
                        print("\n🚨 Anomaly Detected!\nLabel: Unknown\nReason: Failed to retrieve LLM response\n")
                        log_anomaly(data, "Unknown", "Failed to retrieve LLM response", score)
                else:
                    print("✅ Normal traffic.\n")

                # Plot every 50 predictions
                if len(X_buffer) >= 50:
                    print("📊 Plotting PCA for last 50 predictions...\n")
                    plot_pca(X_buffer, label_buffer)
                    X_buffer.clear()
                    label_buffer.clear()

            except json.JSONDecodeError:
                print("Error decoding JSON.")
