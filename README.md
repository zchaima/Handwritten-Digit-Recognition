## MNIST Handwritten Digit Recognition using TensorFlow

This code demonstrates building and training a machine learning model for recognizing handwritten digits using the MNIST dataset with TensorFlow. The MNIST dataset is a popular benchmark dataset consisting of 28x28 pixel grayscale images of handwritten digits (0 through 9).

### Usage

**Dependencies:** Ensure you have the following dependencies installed:

* TensorFlow (for building and training the model)
* NumPy (for mathematical operations)
* Matplotlib (for data visualization)

**Dataset:** The MNIST dataset is used, which is automatically downloaded by TensorFlow. No additional setup is required.

**Training:** Run the script to train the model. It loads the dataset, normalizes the pixel values, explores the data, defines the model architecture, compiles the model, trains it, and evaluates its performance.

**Results:** After training, the script evaluates the model on the test dataset and displays the accuracy and loss. It also makes predictions on a few test images and shows the predicted digit along with the actual image.

### Description of Files

* `mnist_digit_recognition.py`: Python script containing the code for building, training, and evaluating the digit recognition model.

### Code Exploration

**Data Loading and Preprocessing:**

* The MNIST dataset is loaded and normalized to have pixel values between 0 and 1.

**Data Exploration:**

* Some sample images from the training set are displayed along with their labels.
* Additionally, label distribution and image properties are printed for exploration.

**Model Architecture:**

* A simple neural network with:
    * A flattening layer
    * A hidden dense layer with ReLU activation
    * An output layer with softmax activation is defined.

**Model Compilation:**

* The model is compiled with:
    * The Adam optimizer
    * Sparse categorical cross-entropy loss
    * Accuracy metric

**Model Training:**

* The model is trained on the training images and labels for a specified number of epochs.

**Model Evaluation:**

* The trained model is evaluated on the test dataset to measure its performance in terms of accuracy and loss.

**Prediction:**

* Finally, the model makes predictions on a few test images and displays the results.
