import argparse
from pathlib import Path
from typing import List

from config_types import load_config, Config
from data_types import TelescopeConfig, KIDConfig
from tod_preprocessing import TODPreprocessor   # <--- IMPORT DELLA TUA CLASSE


# ------------------------------------------------------------
# Costruttore telescopi
# ------------------------------------------------------------
def build_telescope(telescope_name: str) -> TelescopeConfig:
    name = telescope_name.upper()

    if name == "SRT":
        return TelescopeConfig(
            name="SRT",
            diameter_m=64.0,
            central_freq_hz=90e9,
            bandwidth_hz=30e9,
            efficiency=0.3,
            fov_arcmin=4.0,
        )

    if name == "GBT":
        return TelescopeConfig(
            name="GBT",
            diameter_m=100.0,
            central_freq_hz=43e9,
            bandwidth_hz=8e9,
            efficiency=0.5,
            fov_arcmin=7.0,
        )

    raise ValueError(f"Unknown telescope in a1995_conf.yaml: {telescope_name!r}")


# ------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TOD preprocessing.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/preprocessing/a1995_conf.yaml",
        help="Path to configuration YAML file"
    )
    return parser.parse_args()


# ------------------------------------------------------------
# Load TOD files
# ------------------------------------------------------------
def find_tod_files(cfg: Config) -> List[Path]:
    tod_dir = Path(cfg.paths.tods)

    if not tod_dir.is_dir():
        raise FileNotFoundError(f"TOD directory not found: {tod_dir}")

    # Cambia qui se hai npz, hdf5, ecc.
    files = sorted(tod_dir.glob("*.fits"))

    if not files:
        raise FileNotFoundError(f"No TOD files found in {tod_dir}")

    return files


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def main() -> None:
    # Leggo configurazione
    args = parse_args()
    cfg = load_config(args.config)

    # Creo il modello del telescopio
    telescope = build_telescope(cfg.telescope)

    # Recupero la lista dei TODs
    tod_files = find_tod_files(cfg)

    # Istanzio il preprocessore
    preproc = TODPreprocessor(cfg=cfg, telescope=telescope)


    kid = KIDConfig(
        id="KID_12",
        resonance_freq_hz=4.78e9
    )

    # Preprocesso tutti i file
    for path in tod_files:
        print(f"Processing {path.name} ...")
        preproc.run_one(path)

    print("✔ Preprocessing completed.")


if __name__ == "__main__":
    main()