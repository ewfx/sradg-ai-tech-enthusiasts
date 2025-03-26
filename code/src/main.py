from src.data_preprocessing import preprocess_configs, preprocess_logs
from src.config_reconciliation import reconcile_configs, train_config_embedding_model
from src.log_anomaly_detection import detect_log_anomalies
from src.utils import save_json
import pickle

def main():
    # Configuration Reconciliation
    config1 = preprocess_configs("data/raw/network_configs_day1.txt")
    config2 = preprocess_configs("data/raw/network_configs_day2.txt")

    model, tokenizer = train_config_embedding_model([config1, config2])
    reconciliation_result = reconcile_configs(config1, config2, model, tokenizer)
    save_json(reconciliation_result, "data/processed/reconciled_configs.json")

    # Log Anomaly Detection
    logs1 = preprocess_logs("data/raw/network_logs_day1.csv")
    logs2 = preprocess_logs("data/raw/network_logs_day2.csv")

    anomalies1 = detect_log_anomalies(logs1, train=True)
    save_json(anomalies1.to_json(orient="records"), "data/processed/
