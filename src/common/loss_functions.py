import numpy as np


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray, deriv=False) -> float:
    if deriv:
        return 2 * (predictions - targets) / targets.size
    return np.mean((predictions - targets) ** 2)
