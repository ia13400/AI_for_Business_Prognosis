"""Run univariate future forecasts without unknown future covariates (Experiment 1 models only).

Uses each model's fallback configuration (no HPO) -- a fast, one-shot genuine
future forecast beyond the downloaded data's end, not a rolling evaluation.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from gold_forecasting.config import load_yaml, select_device, set_seed
from gold_forecasting.hashing import dataframe_hash
from gold_forecasting.experiments import run_future
from gold_forecasting.models.sarima import SarimaForecaster
from gold_forecasting.models.patchtst import PatchTSTForecaster
from gold_forecasting.models.chronos_zero_shot import ChronosForecaster
from gold_forecasting.paths import PROCESSED

def main():
    experiment_config = load_yaml("experiments.yaml"); models_config = load_yaml("models.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=[7, 30, 90], default=90)
    parser.add_argument("--mode", choices=["quick_mode", "full_mode"], default=experiment_config["mode"])
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    dataset = args.dataset or max(PROCESSED.glob("gold_dataset_*.parquet"), key=lambda p: p.stat().st_mtime)
    data = pd.read_parquet(dataset); data_hash = dataframe_hash(data)
    hpo = models_config["hpo"][args.mode]; seed = int(models_config["seed"]); set_seed(seed); device = select_device()
    full_series = data["gold_usd"].dropna()

    models = {
        "sarima": SarimaForecaster(config={"order": models_config["sarima"]["fallback"]["order"], "seasonal_period": models_config["sarima"].get("seasonal_period", 0), "retrain_each_step": False}),
        "patchtst": PatchTSTForecaster(config={"context_length": models_config["patchtst"]["context_length"], **models_config["patchtst"]["fallback"], "epochs": hpo["epochs"], "patience": hpo["patience"], "retrain_each_step": False}, device=device, seed=seed),
        "chronos_original": ChronosForecaster(config={"model_id": models_config["chronos"]["variants"]["original"], "context_length": models_config["chronos"].get("context_length", 512)}, device=device),
        "chronos_bolt": ChronosForecaster(config={"model_id": models_config["chronos"]["variants"]["bolt"], "context_length": models_config["chronos"].get("context_length", 512)}, device=device),
    }
    for name, model in models.items():
        forecast = run_future(model, name, full_series, args.horizon, data_hash, seed, {"approach": "univariate-future"}, args.force_retrain)
        print(f"{name}:"); print(forecast.head(10))

if __name__ == "__main__": main()
