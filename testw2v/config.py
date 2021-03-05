import os
import yaml

CONFIG_FILE = 'config.yaml'

def load(config_file=None):
    if config_file is None:
        config_file=CONFIG_FILE
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        except FileNotFoundError as exc:
            return {}

    return config