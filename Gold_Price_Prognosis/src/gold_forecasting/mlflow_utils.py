"""Local MLflow integration with cached-run reuse metadata."""
from contextlib import contextmanager
import json, platform
import mlflow
from .paths import MLFLOW

def configure_mlflow(experiment_type: str) -> None:
    MLFLOW.mkdir(parents=True,exist_ok=True); mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW / 'mlflow.db').resolve().as_posix()}"); mlflow.set_experiment(f"gold-{experiment_type}")

@contextmanager
def tracked_run(model: str, experiment_type: str, tags: dict, cached_run_id: str | None = None):
    configure_mlflow(experiment_type)
    with mlflow.start_run(tags={"model":model,"experiment_type":experiment_type,"cached":str(bool(cached_run_id)).lower(),**{k:str(v) for k,v in tags.items()}}) as run:
        mlflow.log_params({"python":platform.python_version(),"cached_source_run_id":cached_run_id or ""}); yield run

def log_dict_flat(prefix: str, values: dict) -> None:
    mlflow.log_params({f"{prefix}.{k}": json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in values.items()})
