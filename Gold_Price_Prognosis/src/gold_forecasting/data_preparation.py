"""Align market series without filling the gold target."""
import json
from pathlib import Path
import pandas as pd
from .hashing import dataframe_hash, stable_hash
from .paths import PROCESSED, METADATA, ensure_directories

def load_raw(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return frame["value"].sort_index()

def prepare_dataset(target_path: Path, exogenous_paths: dict[str, Path] | None = None, force: bool = False) -> tuple[pd.DataFrame, str]:
    ensure_directories(); target = load_raw(target_path).rename("gold_usd")
    frame = target.to_frame()
    for name, path in (exogenous_paths or {}).items(): frame[name] = load_raw(path).reindex(frame.index).ffill(limit=3)
    frame = frame[~frame.index.duplicated()].sort_index()
    signature = stable_hash({"data": dataframe_hash(frame), "alignment": "target-calendar-exog-ffill-limit-3-v1"})
    output = PROCESSED / f"gold_dataset_{signature[:12]}.parquet"
    manifest = METADATA / f"gold_dataset_{signature[:12]}.json"
    if force or not output.exists(): frame.to_parquet(output)
    manifest.write_text(json.dumps({"signature": signature, "rows": len(frame), "columns": list(frame), "target_fill": "none", "exogenous_fill": "past-only forward fill, maximum 3 target trading dates"}, indent=2), encoding="utf-8")
    return frame, signature
