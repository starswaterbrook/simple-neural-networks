from typing import Callable, Optional

import numpy as np


class Layer:
    def __init__(
        self,
        neuron_count: int,
        activation_function: Callable[[np.ndarray], np.ndarray],
        next_layer_count: Optional[int],
    ):
        self.neuron_count = neuron_count
        self.activation_function = activation_function
        self._weights = self._initialize_weights(next_layer_count)
        self._bias = self._bias = np.zeros((neuron_count,), dtype=np.float64)

    def _initialize_weights(self, next_layer_count: Optional[int]) -> np.ndarray:
        if not next_layer_count:
            weight_matrix = np.random.randn(self.neuron_count, self.neuron_count) * np.sqrt(
                2 / (self.neuron_count + self.neuron_count)
            )
            return weight_matrix

        weight_matrix = np.random.randn(next_layer_count, self.neuron_count) * np.sqrt(
            2 / (next_layer_count + self.neuron_count)
        )
        return weight_matrix

    def __str__(self) -> str:
        return f"<Layer - neuron count: {self.neuron_count}, \
                activation fn: {self.activation_function}>"

    def get_weights(self) -> np.ndarray:
        return self._weights
