"""Run both experiments (univariate + multivariate) over the configured chronological test split."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from gold_forecasting.config import load_yaml, select_device, set_seed
from gold_forecasting.hashing import dataframe_hash
from gold_forecasting.splitting import chronological_split
from gold_forecasting.experiments import run_baselines, compare_results
from gold_forecasting.models.sarima import run_sarima
from gold_forecasting.models.patchtst import run_patchtst
from gold_forecasting.models.chronos_zero_shot import run_chronos_original, run_chronos_bolt
from gold_forecasting.models.sarimax import run_sarimax
from gold_forecasting.models.xgboost_model import run_xgboost
from gold_forecasting.models.tft import run_tft
from gold_forecasting.paths import PROCESSED

def main():
    experiment_config = load_yaml("experiments.yaml"); data_config = load_yaml("data.yaml"); models_config = load_yaml("models.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick_mode", "full_mode"], default=experiment_config["mode"])
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    dataset = args.dataset or max(PROCESSED.glob("gold_dataset_*.parquet"), key=lambda p: p.stat().st_mtime)
    data = pd.read_parquet(dataset); data_hash = dataframe_hash(data)
    hpo = models_config["hpo"][args.mode]; seed = int(models_config["seed"]); set_seed(seed)
    horizons = experiment_config["evaluation_horizons"]; exogenous_names = list(data_config["exogenous"])

    split_uni = chronological_split(data[["gold_usd"]], **experiment_config["split"])
    univariate = {
        "sarima": run_sarima(split_uni.train.gold_usd, split_uni.validation.gold_usd, split_uni.test.gold_usd, models_config["sarima"], hpo, data_hash, seed, args.force_retrain, horizons),
        "patchtst": run_patchtst(split_uni.train.gold_usd, split_uni.validation.gold_usd, split_uni.test.gold_usd, models_config["patchtst"], hpo, data_hash, seed, args.force_retrain, horizons),
        "chronos_original": run_chronos_original(split_uni.train.gold_usd, split_uni.validation.gold_usd, split_uni.test.gold_usd, models_config["chronos"], data_hash, seed, args.force_retrain, horizons),
        "chronos_bolt": run_chronos_bolt(split_uni.train.gold_usd, split_uni.validation.gold_usd, split_uni.test.gold_usd, models_config["chronos"], data_hash, seed, args.force_retrain, horizons),
        **run_baselines(split_uni.train.gold_usd, split_uni.validation.gold_usd, split_uni.test.gold_usd, "univariate", data_hash, seed, {"evaluation_horizons": horizons}, args.force_retrain, experiment_config["moving_average_window"]),
    }
    metrics_uni, _ = compare_results(univariate, "univariate", data_hash)
    print("Experiment 1 (univariate):"); print(metrics_uni.sort_values(["horizon", "mae"]))

    split_multi = chronological_split(data[["gold_usd", *exogenous_names]].dropna(), **experiment_config["split"])
    multivariate = {
        "sarimax": run_sarimax(split_multi.train, split_multi.validation, split_multi.test, models_config["sarimax"], hpo, data_hash, seed, args.force_retrain, horizons),
        "xgboost": run_xgboost(split_multi.train, split_multi.validation, split_multi.test, models_config["xgboost"], hpo, data_config["features"], data_hash, seed, args.force_retrain, horizons),
        "tft": run_tft(split_multi.train, split_multi.validation, split_multi.test, models_config["tft"], hpo, data_hash, seed, args.force_retrain, horizons),
        **run_baselines(split_multi.train.gold_usd, split_multi.validation.gold_usd, split_multi.test.gold_usd, "multivariate", data_hash, seed, {"evaluation_horizons": horizons}, args.force_retrain, experiment_config["moving_average_window"]),
    }
    metrics_multi, _ = compare_results(multivariate, "multivariate", data_hash)
    print("Experiment 2 (multivariate):"); print(metrics_multi.sort_values(["horizon", "mae"]))

if __name__ == "__main__": main()
