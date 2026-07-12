"""Download validated Yahoo Finance series with immutable local raw copies."""
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
import yfinance as yf
from .data_validation import validate_market_data
from .hashing import file_hash
from .paths import RAW, METADATA, ensure_directories

def download_series(
    symbol: str,
    start: str,
    end: str | None = None,
    force_download: bool = False,
    require_positive: bool = False,
) -> Path:
    ensure_directories(); safe = symbol.replace("=", "_").replace("^", "IDX_")
    path = RAW / f"{safe}.csv"; metadata_path = METADATA / f"{safe}.json"
    if path.exists() and metadata_path.exists() and not force_download: return path
    raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    column = "Adj Close" if "Adj Close" in raw else "Close"
    frame = validate_market_data(
        raw[[column]].rename(columns={column: "value"}),
        require_positive=require_positive,
    )
    frame.index.name = "date"; frame.to_csv(path)
    metadata = {"source": "Yahoo Finance", "symbol": symbol, "download_timestamp": datetime.now(timezone.utc).isoformat(),
                "requested_start": start, "requested_end": end, "row_count": len(frame), "columns": list(frame.columns), "sha256": file_hash(path)}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path

def download_all(config: dict, force_download: bool = False) -> list[Path]:
    target = download_series(
        config["target_symbol"],
        config["start"],
        config.get("end"),
        force_download,
        require_positive=True,
    )
    exogenous = [
        download_series(
            symbol,
            config["start"],
            config.get("end"),
            force_download,
            require_positive=False,
        )
        for symbol in config.get("exogenous", {}).values()
    ]
    return [target, *exogenous]
