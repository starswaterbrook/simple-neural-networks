import numpy as np

from src.multi_layer_perceptron import MultiLayerPerceptron

mlp = MultiLayerPerceptron()
mlp.load_model("xor_model.model")

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Test the model
for i, o in zip(inputs, targets):
    print(f"Input: {i}")
    print(f"Output: {np.round(mlp.feed_forward(i), 0)}")
    print(f"Expected: {o}")
    print()