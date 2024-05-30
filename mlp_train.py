import numpy as np

from src.common.activation_functions import ReLU, sigmoid
from src.multi_layer_perceptron import MultiLayerPerceptron

# Because the current MLP is unstable
np.seterr(all="raise")

# Initialize the network
mlp = MultiLayerPerceptron()
mlp.add_layer(2, ReLU)
mlp.add_layer(4, ReLU)
mlp.add_layer(1, sigmoid)

# Training data (XOR problem)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

mlp.train(inputs, targets, epochs=10000, learning_rate=0.01)

# Test the model
for i, o in zip(inputs, targets):
    print(f"Input: {i}")
    print(f"Output: {np.round(mlp.feed_forward(i), 0)}")
    print(f"Expected: {o}")
    print()

mlp.save_model("xor_model")
