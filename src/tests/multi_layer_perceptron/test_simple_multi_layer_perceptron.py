import logging

import numpy as np
import pytest

from src.multi_layer_perceptron import MultiLayerPerceptron

logger = logging.getLogger(__name__)


@pytest.fixture
def simple_mlp_config() -> tuple[MultiLayerPerceptron, np.ndarray, np.ndarray]:
    mlp = MultiLayerPerceptron()

    mlp.add_layer(2, lambda x: x)
    mlp.add_layer(2, lambda x: x)
    mlp.add_layer(2, lambda x: x)

    matrix_0 = np.array([[0.5, -0.3], [0.6, 0.8]])
    matrix_1 = np.array([[-0.2, 0.4], [0.1, 0.9]])

    mlp._layers[0]._weights = matrix_0
    mlp._layers[1]._weights = matrix_1

    return mlp, matrix_0, matrix_1


@pytest.fixture
def simple_test_inputs() -> tuple[np.ndarray]:
    input_vectors = [
        np.array([1, 1]),
        np.array([0, 0]),
        np.array([0.2, 0.8]),
        np.array([0.1, -0.4]),
        np.array([-0.3, -1]),
    ]
    return input_vectors


def test_simple_multi_layer_perceptron(
    simple_mlp_config: tuple[MultiLayerPerceptron, np.ndarray, np.ndarray],
    simple_test_inputs: list[np.ndarray],
):
    mlp, matrix_0, matrix_1 = simple_mlp_config

    for input_vector in simple_test_inputs:
        logger.info(f"Testing MLP for input vector: {input_vector}")
        expected_vector = np.dot(matrix_1, np.dot(matrix_0, input_vector))
        logger.info(f"Expected vector: {expected_vector}")
        output_vector = mlp.feed_forward(input_vector)
        assert output_vector.all() == expected_vector.all()
