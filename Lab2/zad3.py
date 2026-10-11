import numpy as np 

def smart_neuron(weight, expected_outputs, input_data, alpha, epochs):
    for epoch in range(epochs):
        total_error = 0.0
        for i in range(len(input_data)):
            x = input_data[i]
            goal = expected_outputs[i]
            prediction = np.dot(weight, x)
            total_error += np.mean((prediction - goal) **2)
            delta = (2/4) * np.outer(prediction - goal, x)
            weight = weight - alpha * delta
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}, Total Error: {total_error}")
            
    return weight

if __name__ == "__main__":
    inputs = []
    expected_outputs = []
    
    with open("training_colors.txt", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
                
            r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
            color_id = int(parts[3])
            
            inputs.append([r, g, b])
            
            goal = np.zeros(4)
            goal[color_id - 1] = 1
            expected_outputs.append(goal)

    inputs = np.array(inputs)
    expected_outputs = np.array(expected_outputs)
    
    weight = np.random.rand(4, 3) - 0.5
    alpha = 0.01
    epochs = 1000

    final_weights = smart_neuron(weight, expected_outputs, inputs, alpha, epochs)

    correct_answers = 0
    total_tests = 0
    
    with open("test_colors.txt", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
                
            r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
            real_color_id = int(parts[3])
            x = np.array([r, g, b])
            
            prediction = np.dot(final_weights, x)
            
            predicted_color_id = np.argmax(prediction) + 1
            
            total_tests += 1
            if predicted_color_id == real_color_id:
                correct_answers += 1
                
    accuracy = (correct_answers / total_tests) * 100
    print(f"Efficiency of the network on the test set: {accuracy}%")
