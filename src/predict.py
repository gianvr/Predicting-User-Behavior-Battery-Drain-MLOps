import pickle
import pandas as pd


def predict(data:pd.DataFrame)->pd.DataFrame:
    with open("models/user_behavior_model.pkl", 'rb') as file:
        model = pickle.load(file)
    
    predictions = model.predict(data)
    
    return predictions
