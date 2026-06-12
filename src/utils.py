import yaml
import logging


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)