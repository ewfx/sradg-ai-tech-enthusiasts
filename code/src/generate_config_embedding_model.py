# generate_config_embedding_model.py
import torch
from transformers import AutoTokenizer, AutoModel
import pickle

def train_config_embedding_model(model_name="bert-base-uncased"):
    """
    Trains a Gen AI model to generate embeddings for network configurations.

    Args:
        model_name (str): The name of the pre-trained transformer model to use.

    Returns:
        tuple: A tuple containing the trained model and tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Example usage (you might have a list of configs to train on)
        example_configs = [
            "interface GigabitEthernet0/0 ip address 192.168.1.1 255.255.255.0",
            "interface GigabitEthernet0/1 ip address 10.0.0.1 255.255.255.0",
            "router ospf 1 area 0",
            "access-list 101 permit ip any any"
        ]

        # In a real-world scenario, you would fine-tune the model on a larger dataset of network configurations.
        # This example simply initializes the model and tokenizer.

        # Save the model and tokenizer to files
        pickle.dump(model, open("models/config_embedding_model.pkl", 'wb'))
        pickle.dump(tokenizer, open("models/config_tokenizer.pkl", 'wb'))

        print(f"Model and tokenizer saved to models/config_embedding_model.pkl and models/config_tokenizer.pkl")
        return model, tokenizer

    except Exception as e:
        print(f"Error training and saving model: {e}")
        return None, None

if __name__ == "__main__":
    train_config_embedding_model()
