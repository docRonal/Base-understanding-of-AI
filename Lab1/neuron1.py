import numpy as np

def neuron(input, weight, bias):
    output = np.dot(input, weight) + bias 
    print(output)
    return output
neuron(np.array([0.5, 1, 0.5]), np.array([0.2, 0.8, 0.5]), 1)
