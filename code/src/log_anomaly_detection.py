import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

def detect_log_anomalies(logs, model=None, train=False):
    """Detects anomalies in network logs."""
    if train or model is None:
        model = IsolationForest(contamination=0.01)  # Adjust contamination as needed
        model.fit(logs.select_dtypes(include=['number'])) #Train on numerical features
        pickle.dump(model, open("models/log_anomaly_model.pkl", 'wb'))
    else:
        model = pickle.load(open("models/log_anomaly_model.pkl", 'rb'))

    anomalies = model.predict(logs.select_dtypes(include=['number']))
    logs['anomaly'] = anomalies
    return logs[logs['anomaly'] == -1]
