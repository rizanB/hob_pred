import json
import os


def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(current_dir, "config.json")

    with open(config_path, "r") as f:
        return json.load(f)

load_config()