import numpy as np
import json

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class NeuralNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = []
        self.activations = []
        self.layer_sizes = [input_size]

    def add_layer(self, n, weight_min=-0.5, weight_max=0.5, activation=None):
        prev_size = self.layer_sizes[-1]
        w = np.random.uniform(weight_min, weight_max, (n, prev_size))
        self.weights.append(w)
        self.activations.append(activation)
        self.layer_sizes.append(n)

    def predict(self, x):
        current_input = x
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], current_input)
            if self.activations[i] == 'relu':
                current_input = relu(z)
            else:
                current_input = z
        return current_input

    def fit(self, x, expected_output, alpha=0.01):
        layer_inputs = [x]
        layer_outputs = []
        current_input = x
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], current_input)
            layer_outputs.append(z)
            if self.activations[i] == 'relu':
                current_input = relu(z)
            else:
                current_input = z
            layer_inputs.append(current_input)
            
        output = layer_inputs[-1]
        N = len(output)
        delta = (2/N) * (output - expected_output)
        
        deltas = [delta]
        
        for i in range(len(self.weights) - 1, 0, -1):
            w = self.weights[i]
            prev_z = layer_outputs[i-1]
            
            hidden_delta = np.dot(w.T, deltas[-1])
            if self.activations[i-1] == 'relu':
                hidden_delta = hidden_delta * relu_deriv(prev_z)
                
            deltas.append(hidden_delta)
            
        deltas.reverse()
        
        for i in range(len(self.weights)):
            weight_delta = np.outer(deltas[i], layer_inputs[i])
            self.weights[i] -= alpha * weight_delta

    def save_weights(self, file_name):
        data = {
            'input_size': self.input_size,
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'weights': [w.tolist() for w in self.weights]
        }
        with open(file_name, 'w') as f:
            json.dump(data, f)

    def load_weights(self, file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
        self.input_size = data['input_size']
        self.layer_sizes = data['layer_sizes']
        self.activations = data['activations']
        self.weights = [np.array(w) for w in data['weights']]

if __name__ == "__main__":
    inputs = np.array([
        [0.5, 0.75, 0.1],
        [0.1, 0.3, 0.7],
        [0.2, 0.1, 0.6],
        [0.8, 0.9, 0.2]
    ])

    Wh = np.array([
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ])

    Wy = np.array([
        [0.7, 0.9, -0.4, 0.8, 0.1],
        [0.8, 0.5, 0.3, 0.1, 0.0],
        [-0.3, 0.9, 0.3, 0.1, -0.2]
    ])

    nn = NeuralNetwork(input_size=3)
    nn.add_layer(5, activation='relu')
    nn.add_layer(3, activation=None)
    
    nn.weights[0] = Wh
    nn.weights[1] = Wy

    for i in range(len(inputs)):
        output = nn.predict(inputs[i])
        print(f"output {i+1} series:\n{output}\n")