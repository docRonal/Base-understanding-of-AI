# Fundamentals of Artificial Intelligence (Podstawy Sztucznej Inteligencji) 

This repository contains the solutions for Laboratory 1 and Laboratory 2 for the "Fundamentals of Artificial Intelligence" course. The main objective of these labs is to understand the core principles of artificial neural networks by implementing them "from scratch" using the NumPy library.

## 🛠 Technologies

* **Programming Language:** Python 3
* **Libraries:** NumPy (matrix operations, vectorization)

---

## 🔬 Laboratory 1: Neural Network Architecture (Forward Pass)

The first laboratory focuses on understanding the basic components of neural networks and implementing the forward propagation algorithm.

###  Implemented Architectures

**1. The Single Neuron**
Implementation of a basic neuron that calculates an output based on input data, weights, and an optional bias.

```text
   Input (x) ────── weight (W) ─────> [ Σ ] ──────> Prediction (y)
                                        ▲
                                        │
                                      Bias (b)
```

**2. Single-Layer Network (Dense Layer)**
Creation of a fully connected layer where multiple inputs are mapped to multiple outputs using matrix multiplication.

**3. Deep Neural Network**
Implementation of a multi-layer network (including a hidden layer), where the output of the first layer serves as the input for the next.

**4. Modular API**
Development of a flexible class to build networks with an arbitrary number of layers, inputs, and outputs, along with random weight initialization functions.

---

## Laboratory 2: Gradient Descent & Network Training

The second laboratory focuses on neural network training algorithms. We implemented the Gradient Descent method to minimize the Mean Squared Error (MSE).

### Implemented Features

**1. Single Neuron Training**
Implementation of a training loop for a single neuron. This includes calculating the Mean Squared Error (MSE), computing the gradient (Delta), and updating the weight using a specified learning rate ($\alpha$).

**2. Dense Layer Training**
Adapting the gradient descent algorithm for matrix computations. The network is trained on multiple data batches (series) using the outer product to efficiently update the weight matrices.

**3. RGB Color Classification**
Practical application of the network for a classification task. The network takes 3 inputs (R, G, B color channels) and classifies them into 4 distinct colors (Red, Green, Blue, Yellow).

*   **One-Hot Encoding:** Applied to the expected results to format them for classification (e.g., `[0, 0, 0, 1]` for Yellow).
*   **Accuracy Testing:** The model is trained on a training dataset (`training_colors.txt`) and evaluated on a test dataset (`test_colors.txt`). The network achieves a **100% accuracy** rate by selecting the output neuron with the highest activation.

### Visualization of the RGB Classifier (Task 3)

```text
  Input Layer (RGB)     Dense Layer (Weights Matrix 4x3)     Output Layer (Classes)
  
   [x1] (Red)   ───────────────── w_11, w_12... ─────────────────> [y1] (Red)
                                     / \
   [x2] (Green) ───────────────── w_21, w_22... ─────────────────> [y2] (Green)
                                     \ /
   [x3] (Blue)  ───────────────── w_31, w_32... ─────────────────> [y3] (Blue)
   
                                     └───────────────────────────> [y4] (Yellow)
```

---

##  How to Run

Ensure you have Python and NumPy installed on your machine.
Install dependencies if necessary:

```bash
pip install numpy
```

Run the scripts standardly via the terminal. For example, to run the color classifier:

```bash
python lab2_task3.py
```
