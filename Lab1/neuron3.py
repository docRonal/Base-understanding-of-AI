import numpy as np

def deep_neural_network(input, weight_list):

    current_data = input 
    for weight_matrix in weight_list:
        if len(current_data) != weight_matrix.shape[1]:
            return "Error: Input and weight dimensions do not match."
        current_data = np.dot(weight_matrix, current_data)
    return current_data

input = np.array([0.5, 0.75, 0.1])
Wh = np.array([
    [ 0.1,  0.1, -0.3], 
    [ 0.1,  0.2,  0.0],
    [ 0.0,  0.7,  0.1], 
    [ 0.2,  0.4,  0.0], 
    [-0.3,  0.5,  0.1]  
])
Wy = np.array([
    [ 0.7,  0.9, -0.4,  0.8,  0.1], 
    [ 0.8,  0.5,  0.3,  0.1,  0.0],
    [-0.3,  0.9,  0.3,  0.1, -0.2]  
])
result = deep_neural_network(input, [Wh, Wy])
print(result)
