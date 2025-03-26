import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from src.utils import save_json, diff_configs

def generate_config_embeddings(config_text, model, tokenizer):
    """Generates embeddings for network configurations using Gen AI."""
    inputs = tokenizer(config_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def reconcile_configs(config1, config2, model, tokenizer, threshold=0.9):
    """Reconciles network configurations and identifies differences."""

    embedding1 = generate_config_embeddings(config1, model, tokenizer)
    embedding2 = generate_config_embeddings(config2, model, tokenizer)

    similarity = cosine_similarity(embedding1, embedding2)[0][0]

    if similarity < threshold:
        diff = diff_configs(config1, config2)
        return {"similarity": similarity, "difference": diff}
    else:
        return {"similarity": similarity, "difference": "No significant differences found."}

def train_config_embedding_model(configs, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

import pickle
model = pickle.load(open("models/config_embedding_model.pkl", 'rb'))
tokenizer = pickle.load(open("models/config_tokenizer.pkl", 'rb'))
