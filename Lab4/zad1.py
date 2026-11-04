import struct
import numpy as np
import json
import time

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
        dropout_masks = []
        current_input = x
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], current_input)
            layer_outputs.append(z)
            if self.activations[i] == 'relu':
                current_input = relu(z)
                mask = np.random.randint(0, 2, size=current_input.shape)
                current_input = current_input * mask * 2.0
                dropout_masks.append(mask)
            else:
                current_input = z
                dropout_masks.append(np.ones_like(current_input))
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
                hidden_delta = hidden_delta * dropout_masks[i-1]
                
            deltas.append(hidden_delta)
            
        deltas.reverse()
        
        for i in range(len(self.weights)):
            weight_delta = np.outer(deltas[i], layer_inputs[i])
            self.weights[i] -= alpha * weight_delta
            
        return output

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
    test_images = load_mnist_images("mnist/data/MNIST/raw/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("mnist/data/MNIST/raw/t10k-labels-idx1-ubyte")
    
    nn = NeuralNetwork(input_size=784)
    nn.add_layer(100, weight_min=-0.1, weight_max=0.1, activation='relu')
    nn.add_layer(10, weight_min=-0.1, weight_max=0.1, activation=None)
    
    alpha = 0.005
    epochs = 350
    num_train = 10000
    num_test = 10000
    
    start_time = time.time()
    
    for epoch in range(epochs):
        correct_train = 0
        epoch_error = 0.0
        
        for i in range(num_train): 
            x = train_images[i]
            label = train_labels[i]
            
            expected = np.zeros(10)
            expected[label] = 1.0
            
            output = nn.fit(x, expected, alpha=alpha)
            
            epoch_error += np.mean((output - expected) ** 2)
            if np.argmax(output) == label:
                correct_train += 1
                
        if epoch % 10 == 0 or epoch == epochs - 1:
            correct_test = 0
            for i in range(num_test):
                if np.argmax(nn.predict(test_images[i])) == test_labels[i]:
                    correct_test += 1
                    
            train_acc = (correct_train / num_train) * 100
            test_acc = (correct_test / num_test) * 100
            avg_error = epoch_error / num_train
            
            print(f"Iter: {epoch:<3} Error: {avg_error:.6f} Train Accuracy: {correct_train} - {train_acc:.1f}% Test Accuracy: {round(test_acc, 2)}")

    end_time = time.time()
    print(f"Operation took {end_time - start_time:.3f}seconds")
    
    nn.save_weights("mnist_weights.json")