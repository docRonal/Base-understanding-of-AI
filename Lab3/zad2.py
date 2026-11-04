import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

if __name__ == "__main__":
    data_inputs = np.array([
        [0.5, 0.75, 0.1],
        [0.1, 0.3, 0.7],
        [0.2, 0.1, 0.6],
        [0.8, 0.9, 0.2]
    ])

    expected_outputs = np.array([
        [0.1, 0.5, 0.1, 0.7],
        [1.0, 0.2, 0.3, 0.6],
        [0.1, -0.5, 0.2, 0.2]
    ]).T

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

    alpha = 0.01
    epochs = 50
    N = 3

    for epoch in range(epochs):
        if epoch == 0 or epoch == 49:
            print(f"\nOutput at epoch {epoch+1}:")
            
        for i in range(len(data_inputs)):
            x = data_inputs[i]
            goal = expected_outputs[i]

            layer_hidden_1_raw = np.dot(Wh, x)
            layer_hidden_1 = relu(layer_hidden_1_raw)

            layer_output = np.dot(Wy, layer_hidden_1)

            if epoch == 0 or epoch == 49:
                print(f"Output from network: {i+1}, series {layer_output}")
                
            layer_output_delta = (2/N) * (layer_output - goal)
            
            layer_hidden_1_delta = np.dot(Wy.T, layer_output_delta)
            layer_hidden_1_delta = layer_hidden_1_delta * relu_derivative(layer_hidden_1_raw)

            layer_output_weight_delta = np.outer(layer_output_delta, layer_hidden_1)
            layer_hidden_1_weight_delta = np.outer(layer_hidden_1_delta, x)

            Wy = Wy - alpha * layer_output_weight_delta
            Wh = Wh - alpha * layer_hidden_1_weight_delta