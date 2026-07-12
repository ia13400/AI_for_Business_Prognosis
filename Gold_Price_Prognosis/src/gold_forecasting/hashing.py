"""Stable content hashing utilities."""
import hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def canonical_json(value: Any) -> str:
    def default(obj: Any):
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        raise TypeError(type(obj).__name__)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)

def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def dataframe_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    schema = canonical_json({"columns": list(frame.columns), "dtypes": frame.dtypes.astype(str).tolist()})
    return hashlib.sha256(schema.encode() + values).hexdigest()
