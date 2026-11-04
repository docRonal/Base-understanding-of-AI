import struct
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

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows * cols).astype(float) / 255.0
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

if __name__ == "__main__":
    train_images = load_mnist_images("mnist/data/MNIST/raw/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("mnist/data/MNIST/raw/train-labels-idx1-ubyte")
    
    nn = NeuralNetwork(input_size=784)
    nn.add_layer(40, weight_min=-0.1, weight_max=0.1, activation='relu')
    nn.add_layer(10, weight_min=-0.1, weight_max=0.1, activation=None)
    
    print("Start training.")
    epochs = 5
    for epoch in range(epochs):
        correct = 0
        for i in range(1000): 
            x = train_images[i]
            label = train_labels[i]
            
            expected = np.zeros(10)
            expected[label] = 1.0
            
            nn.fit(x, expected, alpha=0.01)
            
            if np.argmax(nn.predict(x)) == label:
                correct += 1
                
        print(f"Epoch {epoch+1}, Accuracy: {correct / 1000 * 100}%")

    nn.save_weights("mnist_weights.json")
    print("All weights saved.")