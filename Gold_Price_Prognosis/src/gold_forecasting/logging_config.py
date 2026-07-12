"""Conservative logging configuration for scripts and notebooks."""
import logging

def configure_logging(level: int = logging.WARNING) -> None:
    logging.basicConfig(level=level,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("mlflow","urllib3","yfinance"): logging.getLogger(noisy).setLevel(logging.WARNING)
