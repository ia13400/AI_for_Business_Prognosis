"""PyTorch sliding-window dataset helpers."""
import numpy as np
import torch
from torch.utils.data import TensorDataset

def windows(values: np.ndarray, context_length: int) -> TensorDataset:
    """Build (context, next-step-target) windows.

    `values` is either 1-D (univariate) or (n, n_features) (multivariate,
    target assumed to be column 0). The predicted step is always the
    target column immediately following each context window.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1: array = array[:, None]
    if len(array) <= context_length: raise ValueError("Series is shorter than context length")
    x = np.stack([array[i-context_length:i] for i in range(context_length, len(array))])
    y = array[context_length:, :1]
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
