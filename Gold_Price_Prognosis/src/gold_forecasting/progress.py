"""Quiet cross-environment progress bars."""
from tqdm.auto import tqdm

def progress(iterable, description: str, total: int | None = None):
    return tqdm(iterable, desc=description, total=total, bar_format="{l_bar}{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]")
