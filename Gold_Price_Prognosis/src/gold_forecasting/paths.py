"""Stable project paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
METADATA = DATA / "metadata"
ARTIFACTS = ROOT / "artifacts"
CACHE = ARTIFACTS / "cache"
CHECKPOINTS = ARTIFACTS / "checkpoints"
PREDICTIONS = ARTIFACTS / "predictions"
METRICS = ARTIFACTS / "metrics"
MANIFESTS = ARTIFACTS / "manifests"
FIGURES = ARTIFACTS / "figures"
MLFLOW = ARTIFACTS / "mlflow"
OPTUNA = ARTIFACTS / "optuna"
HPO_TRIALS = ARTIFACTS / "hpo_trials"
LOSSES = ARTIFACTS / "losses"
FEATURE_IMPORTANCE = ARTIFACTS / "feature_importance"

def ensure_directories() -> None:
    """Create all runtime directories."""
    for path in (RAW, PROCESSED, METADATA, CACHE, CHECKPOINTS, PREDICTIONS, METRICS, MANIFESTS, FIGURES, MLFLOW, OPTUNA, HPO_TRIALS, LOSSES, FEATURE_IMPORTANCE):
        path.mkdir(parents=True, exist_ok=True)
