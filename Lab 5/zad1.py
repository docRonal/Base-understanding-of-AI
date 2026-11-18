import numpy as np

def convolution2d(image, filter_kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
        
    img_h, img_w = image.shape
    kernel_h, kernel_w = filter_kernel.shape
    
    out_h = (img_h - kernel_h) // stride + 1
    out_w = (img_w - kernel_w) // stride + 1
    
    output_image = np.zeros((out_h, out_w))
    
    for y in range(out_h):
        for x in range(out_w):
            y_start = y * stride
            y_end = y_start + kernel_h
            x_start = x * stride
            x_end = x_start + kernel_w
            
            region = image[y_start:y_end, x_start:x_end]
            output_image[y, x] = np.sum(region * filter_kernel)
            
    return output_image

if __name__ == "__main__":
    input_image = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]
    ])
    
    filter_mask = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    
    result = convolution2d(input_image, filter_mask, stride=1, padding=0)
    print("Output:\n", result)
