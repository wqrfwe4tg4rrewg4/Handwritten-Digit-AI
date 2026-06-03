# Handwritten-Digit-AI

A handwritten digit recognition application built from scratch using Python and NumPy.

The project demonstrates how a feed-forward neural network works internally without relying on high-level machine learning frameworks for training. The model is trained on the MNIST dataset and can recognize digits drawn by the user in a graphical interface.

---

## Features

* Neural network implemented from scratch
* Training on the MNIST dataset
* Real-time handwritten digit recognition
* Interactive drawing interface
* Automatic saving and loading of trained weights
* Educational implementation of forward and backward propagation

---

## Neural Network Architecture

Input Layer → Hidden Layer → Output Layer

```text
784 → 64 → 10
```

Where:

* 784 inputs represent a 28×28 image
* 64 hidden neurons learn image patterns
* 10 output neurons represent digits from 0 to 9

---

## How It Works

### Training

The neural network is trained on the MNIST dataset.

Each image:

1. Is converted into a vector of 784 pixel values.
2. Passes through the hidden layer.
3. Produces probabilities for digits 0–9.
4. Calculates prediction error.
5. Updates weights using backpropagation.

After training, the learned weights are stored in `.npy` files.

---

### Prediction

The user draws a digit inside the application.

The image is:

1. Converted to grayscale.
2. Resized to 28×28 pixels.
3. Normalized to match MNIST format.
4. Passed through the trained neural network.

The application then displays:

* Predicted digit
* Confidence score

---

## Project Structure

```text
main.py              # Neural network training
draw_predict.py      # Drawing application
utils.py             # Dataset loading
weights_i_h.npy      # Input → Hidden weights
weights_h_o.npy      # Hidden → Output weights
bias_i_h.npy         # Hidden layer biases
bias_h_o.npy         # Output layer biases
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Handwritten-Digit-AI.git
cd Handwritten-Digit-AI
```

Install dependencies:

```bash
pip install numpy pillow tensorflow
```

---

## Training the Model

Run:

```bash
python main.py
```

The model will train on MNIST and generate weight files.

---

## Running the Application

Run:

```bash
python draw_predict.py
```

Draw a digit and click **Predict**.

---

## Dataset

This project uses the MNIST handwritten digit dataset:

* 60,000 training images
* 10,000 testing images
* Digits 0–9

---

## Example Accuracy

Typical training accuracy:

```text
97–98%
```

---

## Future Improvements

* Letter recognition using EMNIST
* Convolutional Neural Networks (CNN)
* Improved image preprocessing
* Better drawing interface
* Exporting trained models

---

## License

This project is provided for educational purposes.
