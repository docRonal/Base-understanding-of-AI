import struct
import numpy as np
import time

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def get_image_regions(image):
    regions = np.zeros((26 * 26, 9))
    idx = 0
    for y in range(26):
        for x in range(26):
            regions[idx] = image[y:y+3, x:x+3].flatten()
            idx += 1
    return regions

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        return image_data.reshape(num_images, rows, cols).astype(float) / 255.0 

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

if __name__ == "__main__":
    train_images = load_mnist_images("mnist/data/MNIST/raw/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("mnist/data/MNIST/raw/train-labels-idx1-ubyte")
    test_images = load_mnist_images("mnist/data/MNIST/raw/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("mnist/data/MNIST/raw/t10k-labels-idx1-ubyte")
    
    W_conv = np.random.uniform(-0.01, 0.01, (16, 9))
    W_fc = np.random.uniform(-0.1, 0.1, (10816, 10))
    
    alpha = 0.01
    epochs = 50
    num_train = 10000 #  10000 or 60000
    num_test = 10000
    
    start_time = time.time()
    
    for epoch in range(epochs):
        correct_train = 0
        
        for i in range(num_train): 
            x = train_images[i]
            label = train_labels[i]
            y_true = np.zeros((1, 10))
            y_true[0, label] = 1.0
            
            # forward pass
            X_col = get_image_regions(x) 
            
            Z_conv = np.dot(X_col, W_conv.T) 
            A_conv = relu(Z_conv)
            
            # flatten 
            A_conv_flat = A_conv.flatten().reshape(1, -1) 
            
            # fully connected layer
            Z_fc = np.dot(A_conv_flat, W_fc) 
            A_fc = softmax(Z_fc)
            
            if np.argmax(A_fc) == label:
                correct_train += 1
                
            # backward pass
            dZ_fc = A_fc - y_true 
            
            # gradient
            dW_fc = np.dot(A_conv_flat.T, dZ_fc) 
            
            
            dA_conv_flat = np.dot(dZ_fc, W_fc.T) 
            dA_conv = dA_conv_flat.reshape(676, 16)
            
            #gradient for convolutional layer
            dZ_conv = dA_conv * relu_deriv(Z_conv) 
            
            # gradient for convolutional weights
            dW_conv = np.dot(dZ_conv.T, X_col) 
            
            # update weights
            W_fc -= alpha * dW_fc
            W_conv -= alpha * dW_conv

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {correct_train/num_train*100:.2f}%")

    print(f"Training time {time.time() - start_time:.2f} сек.")