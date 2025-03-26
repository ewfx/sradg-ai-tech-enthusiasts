import json
import difflib

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def diff_configs(config1, config2):
    """Generates a human-readable diff of two configuration strings."""
    diff = difflib.unified_diff(config1.splitlines(), config2.splitlines())
    return '\n'.join(diff)
