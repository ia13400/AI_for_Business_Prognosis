"""Resumable Optuna studies backed by local SQLite storage.

Each study is keyed by a name that should embed a data/config signature, so
an interrupted run resumes exactly where it left off: only the trials still
missing from the study's SQLite database are executed on the next call.
"""
from pathlib import Path
from typing import Callable
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_study(study_name: str, objective: Callable[[optuna.Trial], float], n_trials: int, storage_dir: Path,
              seed: int = 42, direction: str = "minimize", timeout: float | None = None) -> optuna.Study:
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(storage_dir / f'{study_name}.db').resolve().as_posix()}"
    study = optuna.create_study(study_name=study_name, storage=storage, direction=direction, load_if_exists=True, sampler=TPESampler(seed=seed))
    completed = sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, n_trials - completed)
    if remaining: study.optimize(objective, n_trials=remaining, timeout=timeout, show_progress_bar=False)
    return study
