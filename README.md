# Fundamentals of Artificial Intelligence

This repository contains Python implementations of neural networks and training methods developed for four artificial intelligence laboratories. The examples use NumPy and progress from forward propagation to backpropagation, MNIST classification, and dropout regularization.

## Requirements

- Python 3
- NumPy
- Torchvision for downloading the MNIST dataset in Lab3

Install the packages with:

```bash
python -m pip install numpy torchvision
```

## Laboratory 1: Neural Network Architecture

Lab1 introduces the building blocks of a feed-forward neural network: a single neuron, a dense layer, multi-layer networks, and a modular network API. The implementations focus on calculating predictions with the forward pass.

Scripts: `Lab1/neuron_1.py` through `Lab1/neuron_4.py`.

## Laboratory 2: Gradient Descent and Training

Lab2 adds training with gradient descent and mean squared error. The exercises cover updating a single neuron and dense-layer weights, then apply the network to four-class RGB color classification. The classifier uses one-hot targets and reports accuracy on separate training and test datasets.

Run the color-classification task from its directory so the data files are found:

```bash
cd Lab2
python zad3.py
```

The datasets are `Lab2/training_colors.txt` and `Lab2/test_colors.txt`.

## Laboratory 3: Backpropagation

Lab3 implements backpropagation for multi-layer networks, first with a small example and then in training tasks. It includes MNIST digit classification and an RGB color classifier.

- `zad1.py` demonstrates predictions through a two-layer network.
- `zad2.py` calculates gradients and updates weights for a hidden-layer network.
- `zad3.py` trains a network on MNIST and saves weights to `mnist_weights.json`.
- `zad4.py` trains and evaluates a four-class RGB color classifier.

Run a script from the Lab3 directory, for example:

```bash
cd Lab3
python zad2.py
```

`mnist_downloader.py` downloads MNIST using Torchvision. Note that it writes data under `Lab3/data/MNIST/raw`, while `zad3.py` currently expects files under `Lab3/mnist/data/MNIST/raw`. Place the downloaded files at the path expected by the script before running `zad3.py`.

## Laboratory 4: Dropout and Mini-Batch Training

Lab4 explores dropout regularization and training on MNIST. The tasks compare single-example and mini-batch updates, and experiment with ReLU, tanh, and softmax activations. Training logs are stored in `Lab4/Lab4_1` and `Lab4/Lab4_2`; the MNIST data files are under `Lab4/mnist/data/MNIST/raw`.

Run the experiments from the Lab4 directory:

```bash
cd Lab4
python zad1.py
python zad2.py
python zad3.py
```

The scripts save trained weights as `mnist_weights.json` in the current directory.
