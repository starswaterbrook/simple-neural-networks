import numpy as np

from src.common.activation_functions import ReLU, sigmoid
from src.multi_layer_perceptron import MultiLayerPerceptron

# Because the current MLP is unstable, if running into numpy issues change the training parameters
np.seterr(all="raise")

# Initialize the network
mlp = MultiLayerPerceptron()
mlp.add_layer(784, ReLU)
mlp.add_layer(20, ReLU)
mlp.add_layer(20, ReLU)
mlp.add_layer(10, sigmoid)


def load_csv_to_numpy_list(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    designated_values = data[:, 0]
    array_values = data[:, 1:]

    result = [
        (designated_value, array_value)
        for designated_value, array_value in zip(designated_values, array_values)
    ]

    return result


filename = "example_usage/train.csv"
numpy_list = load_csv_to_numpy_list(filename)

inputs = []
outputs = []


def normalize_array(arr):
    arr = arr.astype(np.float32)

    normalized_arr = arr / 255.0

    return normalized_arr


for item in numpy_list:
    inputs.append(normalize_array(item[1]))

    output = [0.0 for _ in range(10)]
    output[int(item[0])] = 1.0
    outputs.append(np.array(output))

mlp.train(inputs, outputs, 100, learning_rate=0.005)

mlp.save_model("digit_recognizer")

# The accuracy is measured on the same data, in normal usage it should be split into training and eval data
correct = 0
for i, o in zip(inputs, outputs):
    output = np.round(mlp.feed_forward(i))
    max_idx_guess = np.argmax(output)
    max_idx_correct = np.argmax(o)
    if max_idx_guess == max_idx_correct:
        correct += 1
print(f"Accuracy: {correct/len(inputs)}")
