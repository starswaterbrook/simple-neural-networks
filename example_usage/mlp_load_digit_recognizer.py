import numpy as np

from src.multi_layer_perceptron import MultiLayerPerceptron

mlp = MultiLayerPerceptron.load_model("digit_recognizer.model")

def load_csv_to_numpy_list(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    designated_values = data[:, 0]
    array_values = data[:, 1:]

    result = [(designated_value, array_value) for designated_value, array_value in zip(designated_values, array_values)]

    return result

filename = 'example_usage/train.csv'
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

# Test the model
correct = 0
for i, o in zip(inputs, outputs):
    output = np.round(mlp.feed_forward(i))
    max_idx_guess = np.argmax(output)
    max_idx_correct = np.argmax(o)
    if max_idx_guess == max_idx_correct:
        correct += 1
print(f"Accuracy: {correct/len(inputs)}")
