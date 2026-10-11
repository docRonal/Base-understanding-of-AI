import numpy as np

def smart_neuron(weight, goal , input_data, alpha, epochs):
    for i in range(epochs):
        prediction = input_data * weight
        error = (prediction - goal) **2
        delta = 2 * (prediction - goal) *input_data
        weight = weight - alpha * delta
        print(f"Epach: {i+1}, Weight: {weight}, Error: {error}, Prediction: {prediction}")
    return (weight, error, prediction)

if __name__ == "__main__":
    smart_neuron(0.5,0.8,2,0.1, 20)
