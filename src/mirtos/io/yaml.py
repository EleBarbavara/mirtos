import yaml
from pathlib import Path

def to_yaml(file_path: Path):
    ...

def read_yaml(file_path: Path):

    with open(file_path, 'r') as file:
        return yaml.safe_load(file)