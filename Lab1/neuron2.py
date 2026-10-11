import numpy as np

def neural_network(input, weight):

    if len(input) != weight.shape[1]:
        return "Error: Input and weight dimensions do not match."

    output = np.dot(weight, input)
    return output

input = np.array([0.5, 0.75, 0.1])
weight = np.array([
    [ 0.1,  0.1, -0.3], 
    [ 0.1,  0.2,  0.0], 
    [ 0.0,  0.7,  0.1], 
    [ 0.2,  0.4,  0.0], 
    [-0.3,  0.5,  0.1]  
])
result = neural_network(input, weight)
    
print (result)
