import logging
import os
from datetime import datetime

def setup_logger(log_dir, name="scilake"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        ch = logging.StreamHandler()

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"Logger initialized — writing to {log_path}")
    return logger