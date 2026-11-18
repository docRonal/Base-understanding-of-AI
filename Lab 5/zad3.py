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

# Max Pooling 2x2
def max_pool_forward(conv_out):
    pool_out = np.zeros((13, 13, 16))
    max_indices = np.zeros((13, 13, 16, 2), dtype=int)
    for c in range(16):
        for y in range(13):
            for x in range(13):
                y_start, x_start = y * 2, x * 2
                patch = conv_out[y_start:y_start+2, x_start:x_start+2, c]
                pool_out[y, x, c] = np.max(patch)
                #index of the max value in the 2x2 patch
                max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                max_indices[y, x, c] = [y_start + max_idx[0], x_start + max_idx[1]]
    return pool_out, max_indices

# Max Pooling Backward
def max_pool_backward(d_pool_out, max_indices):
    d_conv_out = np.zeros((26, 26, 16))
    for c in range(16):
        for y in range(13):
            for x in range(13):
                max_y, max_x = max_indices[y, x, c]
                d_conv_out[max_y, max_x, c] += d_pool_out[y, x, c]
    return d_conv_out

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
    

    W_conv = np.random.uniform(-0.01, 0.01, (16, 9))

    W_fc = np.random.uniform(-0.1, 0.1, (2704, 10))
    
    alpha = 0.001  
    epochs = 300   
    num_train = 1000 
    
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
            
            A_conv_3d = A_conv.reshape(26, 26, 16)
            A_pool, max_indices = max_pool_forward(A_conv_3d)
            
            # Flatten after pooling
            A_pool_flat = A_pool.flatten().reshape(1, -1) 
            
            Z_fc = np.dot(A_pool_flat, W_fc) 
            A_fc = softmax(Z_fc)
            
            if np.argmax(A_fc) == label:
                correct_train += 1
                
            # backward pass
            dZ_fc = A_fc - y_true 
            dW_fc = np.dot(A_pool_flat.T, dZ_fc) 
            
            dA_pool_flat = np.dot(dZ_fc, W_fc.T) 
            dA_pool = dA_pool_flat.reshape(13, 13, 16)
            
            # error backpropagation through max pooling
            dA_conv_3d = max_pool_backward(dA_pool, max_indices)
            dA_conv = dA_conv_3d.reshape(676, 16)
            
            dZ_conv = dA_conv * relu_deriv(Z_conv) 
            dW_conv = np.dot(dZ_conv.T, X_col) 
            
            # update weights
            W_fc -= alpha * dW_fc
            W_conv -= alpha * dW_conv

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {correct_train/num_train*100:.2f}%")

    print(f"Training time: {time.time() - start_time:.2f} сек.")