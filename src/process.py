import pandas as pd
import os
from kagglehub import kagglehub


def process_data()->None:
    if not os.path.exists("data/ev_charging_patterns.csv"):
        df = pd.read_csv(kagglehub.dataset_download("valakhorasani/mobile-device-usage-and-user-behavior-dataset", path='user_behavior_dataset.csv'))
        df.to_csv("data/user_behavior_dataset.csv", index=False)
    else:
        df = pd.read_csv("data/user_behavior_dataset.csv")

    df.drop(['User ID'], axis=1, inplace=True)

    df.to_csv("data/user_behavior_dataset_processed.csv", index=False)
   

if __name__ == "__main__":
    process_data()
