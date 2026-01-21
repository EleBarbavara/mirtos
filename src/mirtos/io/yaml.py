import yaml
import dacite
from pathlib import Path
import astropy.units as u

def to_yaml(file_path: Path):
    ...

def read_yaml(file_path: Path):

    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# funzione generica che utilizziamo nelle dataclasses per non ripetere lo stesso codice
def from_yaml(cls, filename: Path):
    """
    Creates an instance of the class from data stored in a YAML file.

    This method reads a YAML file, parses its content, and maps the data
    to the fields in the dataclass using dacite. If any attributes of the
    dataclass are of specific types (e.g., `u.Quantity`), dacite will cast
    those attributes appropriately during the instantiation process.

    Args:
        filename (Path): Path to the YAML file containing configuration data.

    Returns:
        An instance of the class (cls) populated with data from the YAML file.
    """


    cfg = read_yaml(filename)


    # dacite.Config(cast=[u.Quantity], dato un attributo della dataclass del tipo u.Quantity
    # dacite richiama il costruttore u.Quantity sul valore letto nel file yaml associato all'attributo
    return dacite.from_dict(data_class=cls,
                            data=cfg,
                            config=dacite.Config(cast=[u.Quantity]))