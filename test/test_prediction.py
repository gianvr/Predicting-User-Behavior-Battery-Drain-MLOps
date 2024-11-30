import requests
import json

data = {
    "dataframe_split": {
        "columns": [
            "Device Model",
            "Operating System",
            "App Usage Time (min/day)",
            "Screen On Time (hours/day)",
            "Battery Drain (mAh/day)",
            "Number of Apps Installed",
            "Data Usage (MB/day)",
            "Age",
            "Gender"
        ],
        "data": [
            [
                "Google Pixel 5",
                "Android",
                393,
                6.4,
                1872,
                67,
                1122,
                40,
                "Male"
            ]
        ]
    }
}


json_data = json.dumps(data)

mlflow_url = "http://127.0.0.1:8080/invocations"

headers = {"Content-Type": "application/json"}
response = requests.post(mlflow_url, headers=headers, data=json_data)

print("Resposta do modelo:", response.text)
