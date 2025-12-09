import yaml

def to_yaml(file_path: str):
    ...

def read_yaml(file_path: str):

    with open(file_path, 'r') as file:
        return yaml.safe_load(file)