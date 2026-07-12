"""PyTorch sliding-window dataset helpers."""
import numpy as np
import torch
from torch.utils.data import TensorDataset

def windows(values: np.ndarray, context_length: int) -> TensorDataset:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(array) <= context_length: raise ValueError("Series is shorter than context length")
    x = np.stack([array[i-context_length:i] for i in range(context_length, len(array))])[:, :, None]
    y = array[context_length:, None]
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
