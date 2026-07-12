"""Download and prepare all configured market data."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gold_forecasting.config import load_yaml
from gold_forecasting.data_download import download_all
from gold_forecasting.data_preparation import prepare_dataset

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--force-download",action="store_true"); args=parser.parse_args(); config=load_yaml("data.yaml")
    paths=download_all(config,args.force_download); names=list(config["exogenous"]); frame,signature=prepare_dataset(paths[0],dict(zip(names,paths[1:])),args.force_download); print(f"Prepared {len(frame)} rows ({signature[:12]}).")
if __name__=="__main__": main()
