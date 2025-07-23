# client.py: Receives streaming network traffic data, detects anomalies using Isolation Forest,
# and generates LLM-based alerts using Together AI API. Logs anomalies to a CSV file.

import socket
import json
import pandas as pd
import joblib
import os
import csv
import datetime
from together import Together

HOST = 'localhost'
PORT = 9999

# Load the trained model and preprocessor
model = joblib.load("anomaly_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

# Initialize Together AI client
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Initialize CSV file for logging anomalies
if not os.path.exists('anomalies.csv'):
    with open('anomalies.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Data', 'Label', 'Reason'])

def pre_process_data(data):
    # Convert data to DataFrame for model prediction
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

def log_anomaly(data, label, reason):
    # Log anomaly details to CSV
    with open('anomalies.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now(), str(data), label, reason])

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

                # Preprocess the data
                X = pre_process_data(data)
                # Predict anomaly (1 = normal, -1 = anomaly)
                prediction = model.predict(X)
                is_anomaly = 1 if prediction[0] == -1 else 0

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
                        log_anomaly(data, label, reason)

                    except Exception as e:
                        print(f"Error with Together AI API: {e}")
                        print("\n🚨 Anomaly Detected!\nLabel: Unknown\nReason: Failed to retrieve LLM response\n")
                        log_anomaly(data, "Unknown", "Failed to retrieve LLM response")

            except json.JSONDecodeError:
                print("Error decoding JSON.")
