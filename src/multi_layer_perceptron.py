import os
import pickle
from typing import Callable, Optional

import numpy as np

from src.common.loss_functions import mean_squared_error
from src.common.neuron_layer import Layer


class MultiLayerPerceptron:
    def __init__(self):
        self._layers: list[Layer] = []

    def add_layer(self, neuron_count: int, activation_function: Callable[[np.ndarray], np.ndarray]):
        new_layer = Layer(neuron_count, activation_function, None)

        if not self.layer_count:
            self._layers.append(new_layer)
            return

        self._layers.append(new_layer)
        self._restructure_layers()

    def remove_last_layer(self):
        self._layers = self._layers[:-1]

    def change_layer(
        self,
        layer_idx: int,
        new_neuron_count: Optional[int],
        new_activation_function: Callable[[np.ndarray], np.ndarray],
    ):
        if new_neuron_count:
            self._layers[layer_idx].neuron_count = new_neuron_count
        if new_activation_function:
            self._layers[layer_idx].activation_function = new_activation_function
        if new_neuron_count or new_activation_function:
            self._restructure_layers()

    def get_layer(self, layer_idx: int) -> Layer:
        return self._layers[layer_idx]

    def feed_forward(self, vector: np.ndarray) -> np.ndarray:
        a = vector.copy()

        for i in range(0, len(self._layers) - 1):
            layer = self._layers[i]
            next_layer = self._layers[i + 1]

            activation_fn = layer.activation_function
            weights = layer.get_weights()
            bias = next_layer._bias

            a = activation_fn(np.dot(weights, a)) + bias

        a = self._layers[-1].activation_function(a)
        return a

    def save_model(self, filename: str):
        model_data = {
            "layers": [
                {
                    "neuron_count": layer.neuron_count,
                    "weights": layer.get_weights(),
                    "bias": layer._bias,
                    "activation_fn": layer.activation_function,
                }
                for layer in self._layers
            ]
        }
        os.makedirs("models", exist_ok=True)
        with open(f"models/{filename}.model", "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filename: str):
        with open(f"models/{filename}", "rb") as f:
            model_data = pickle.load(f)
        self._layers = []
        for i, layer_data in enumerate(model_data["layers"]):
            self.add_layer(layer_data["neuron_count"], layer_data["activation_fn"])
            self._layers[i]._weights = layer_data["weights"]
            self._layers[i]._bias = layer_data["bias"]

    def _get_activations(self, vector: np.ndarray) -> list[np.ndarray]:
        activations = [vector]

        a = vector.copy()
        for i in range(len(self._layers) - 1):
            layer = self._layers[i]
            next_layer = self._layers[i + 1]

            activation_fn = layer.activation_function
            weights = layer.get_weights()
            bias = next_layer._bias

            a = activation_fn(np.dot(weights, a)) + bias
            activations.append(a)

        a = self._layers[-1].activation_function(a)
        activations.append(a)

        return activations

    def _restructure_layers(self):
        layer_neuron_counts = [layer.neuron_count for layer in self._layers]
        layer_activation_fns = [layer.activation_function for layer in self._layers]
        layer_neuron_counts.append(0)
        self._layers = []

        for i in range(len(layer_neuron_counts) - 1):
            layer = Layer(
                layer_neuron_counts[i],
                layer_activation_fns[i],
                layer_neuron_counts[i + 1],
            )
            self._layers.append(layer)

    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int, learning_rate: float):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(inputs, targets):
                activations = self._get_activations(x)

                predictions = activations[-1]
                loss = mean_squared_error(predictions, y)
                total_loss += loss
                loss_derivative = mean_squared_error(predictions, y, deriv=True)

                delta = loss_derivative * self._layers[-1].activation_function(
                    predictions, deriv=True
                )
                for i in reversed(range(1, self.layer_count)):
                    prev_activation = activations[i - 1]
                    next_layer = self._layers[i - 1]
                    current_layer = self._layers[i]
                    current_layer._bias -= learning_rate * delta
                    next_layer._weights -= learning_rate * np.outer(delta, prev_activation)
                    if i > 1:
                        delta = np.dot(next_layer.get_weights().T, delta) * self._layers[
                            i - 1
                        ].activation_function(activations[i - 1], deriv=True)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(inputs)}")

    @property
    def layer_count(self):
        return len(self._layers)

    @property
    def input_size(self):
        return self._layers[0].neuron_count

    @property
    def output_size(self):
        return self._layers[-1].neuron_count
