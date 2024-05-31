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

        for i in range(len(self._layers)-1):
            layer = self._layers[i]

            next_layer = self._layers[i + 1]
            bias = next_layer._bias
            activation_fn = next_layer.activation_function

            weights = layer.get_weights()
            a = activation_fn(np.dot(weights, a) + bias)

        return a

    def save_model(self, filename: str):
        os.makedirs("models", exist_ok=True)
        with open(f"models/{filename}.model", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filename: str):
        with open(f"models/{filename}", "rb") as f:
            model = pickle.load(f)
        if isinstance(model, cls):
            return model
        raise TypeError(f"Loaded model is not of type {cls}")

    def _get_activations(self, vector: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [vector]
        pre_activations = [vector]

        a = vector.copy()
        for i in range(len(self._layers)-1):
            layer = self._layers[i]
            next_layer = self._layers[i + 1]

            bias = next_layer._bias
            activation_fn = next_layer.activation_function
            weights = layer.get_weights()
            
            a = np.dot(weights, a) + bias
            activations.append(activation_fn(a))
            pre_activations.append(a)

        return activations, pre_activations

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
                activations, _ = self._get_activations(x)
                predictions = self.feed_forward(x)

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
