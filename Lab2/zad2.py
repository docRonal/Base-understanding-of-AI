import numpy as np 

def smart_neuron(weight, goal , input_data, alpha, epochs):
    for epoch in range(epochs):
        total_error = 0.0
        for i in range(len(input_data)):
            x = input_data[i]
            goal = expected_outputs[i]
            prediction = np.dot(weight, x)
            total_error += np.mean((prediction - goal) **2)
            delta = (2/5) * np.outer(prediction - goal,x)
            weight = weight - alpha * delta
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}, Total Error: {total_error}")
            
    return weights

if __name__ == "__main__":
    weights = np.array([
        [ 0.1,  0.1, -0.3],
        [ 0.1,  0.2,  0.0],
        [ 0.0,  0.7,  0.1],
        [ 0.2,  0.4,  0.0],
        [-0.3,  0.5,  0.1]
    ])
    inputs = np.array([
        [0.5,  0.75, 0.1],
        [0.1,  0.3,  0.7],
        [0.2,  0.1,  0.6],
        [0.8,  0.9,  0.2]
    ])
    expected_outputs = np.array([
        [ 0.1,  1.0,  0.1,  0.0, -0.1],
        [ 0.5,  0.2, -0.5,  0.3,  0.7],
        [ 0.1,  0.3,  0.2,  0.9,  0.1],
        [ 0.7,  0.6,  0.2, -0.1,  0.8]
    ])
    
    alpha = 0.01
    epochs = 1000
    
    final_weights = smart_neuron(weights, expected_outputs, inputs, alpha, epochs)
