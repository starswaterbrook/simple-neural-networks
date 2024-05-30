import numpy as np


def ReLU(x: np.ndarray, deriv=False) -> np.ndarray:
    if deriv:
        return np.where(x > 0, 1.0, 0.0)
    return np.maximum(0, x)


def softmax(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    max_x = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - max_x)
    if deriv:
        _softmax_deriv(e_x / np.sum(e_x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _softmax_deriv(softmax_output: np.ndarray) -> np.ndarray:
    jacobian_m = np.zeros((softmax_output.shape[0], softmax_output.shape[0]))
    for i in range(softmax_output.shape[0]):
        for j in range(softmax_output.shape[0]):
            if i == j:
                jacobian_m[i][j] = softmax_output[i] * (1 - softmax_output[i])
            else:
                jacobian_m[i][j] = -softmax_output[i] * softmax_output[j]
    return jacobian_m


def sigmoid(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    def safe_exp(x):
        return np.exp(np.clip(x, -500, 500))

    if deriv:
        sig = 1.0 / (1.0 + np.exp(-x))
        return sig * (1.0 - sig)
    return 1.0 / (1.0 + safe_exp(-x))
