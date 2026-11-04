import numpy as np

class Neural_network:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weight = []

    def add_layer(self, n, weight_range=[-1.0, 1.0]):
        if len(self.weight) == 0:
            input_for_this_layaer = self.input_size
        else:
             input_for_this_layaer = self.weight[-1].shape[0]
        min_val,max_val = weight_range
        new_matrix = np.random.uniform(min_val, max_val,(n, input_for_this_layaer))
        self.weight.append(new_matrix)

    def predict(self,input):
        current_data = np.array(input) 
        for weight_matrix in self.weight:
                if len(current_data) != weight_matrix.shape[1]:
                    return "Error: Input and weight dimensions do not match."
                current_data = np.dot(weight_matrix, current_data)
        return current_data    
    
    def load_weights(self, file_name):
        try:
             self.weight = np.load(file_name, allow_pickle=True).tolist()
        except Exception as e:
            print(f"Error loading weights from {file_name}: {e}")

    def save_weights(self, file_name):
        try:
            np.save(file_name, np.array(self.weight, dtype=object))
        except Exception as e:
            print(f"Error saving weights to {file_name}: {e}")

my_net = Neural_network(input_size=3)

my_net.add_layer(n=5, weight_range=[-0.5, 0.5])

my_net.add_layer(n=3, weight_range=[-1.0, 1.0])

inputs = [0.5, 0.75, 0.1]
result = my_net.predict(inputs)
print("Result:", result)


my_net.save_weights("my_model_weights")
my_net.load_weights("my_model_weights.npy")
