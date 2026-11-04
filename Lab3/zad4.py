import numpy as np

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

def load_color_data(filename):
    inputs, outputs, real_ids = [], [], []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if not parts: continue
            
            r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
            color_id = int(parts[3])
            
            inputs.append([r, g, b])
            real_ids.append(color_id)
            
            goal = np.zeros(4)
            goal[color_id - 1] = 1
            outputs.append(goal)
            
    return np.array(inputs), np.array(outputs), real_ids

if __name__ == "__main__":
    train_inputs, train_outputs, train_real_ids = load_color_data("training_colors.txt")
    test_inputs, _, test_real_ids = load_color_data("test_colors.txt")

    nn = NeuralNetwork(input_size=3)
    
    nn.add_layer(5, weight_min=-0.5, weight_max=0.5, activation='relu')
    
    nn.add_layer(4, weight_min=-0.5, weight_max=0.5, activation=None)

    alpha = 0.01
    epochs = 1500

    print("Starting training")
    for epoch in range(epochs):
        for i in range(len(train_inputs)):
            nn.fit(train_inputs[i], train_outputs[i], alpha)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            correct_train = 0
            for i in range(len(train_inputs)):
                prediction = nn.predict(train_inputs[i])
                if np.argmax(prediction) + 1 == train_real_ids[i]:
                    correct_train += 1
            train_acc = (correct_train / len(train_inputs)) * 100
            
            correct_test = 0
            for i in range(len(test_inputs)):
                prediction = nn.predict(test_inputs[i])
                if np.argmax(prediction) + 1 == test_real_ids[i]:
                    correct_test += 1
            test_acc = (correct_test / len(test_inputs)) * 100
            
            print(f"Epoch {epoch + 1:4d} | Accuracy (Train): {train_acc:6.2f}% | Accuracy (Test): {test_acc:6.2f}%")