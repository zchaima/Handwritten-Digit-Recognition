import tensorflow as tf #building and training machine learning models , where our dataset is located
import numpy as np #mathematical operations
import matplotlib.pyplot as plt #DataVisualisation

def main():
    # Load the MNIST dataset of handwritten digits  
    (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Normalization ensures : stabilizing and speeding up the training process.
    # Normalize the pixel values from their original range (0 to 255) to be between 0 and 1
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # Data Exploration
    # Display the first 9 training images with their labels
    plt.figure(figsize=(9, 3))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(training_images[i], cmap='gray')
        plt.title(f"Label: {training_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Check label distribution
    unique, counts = np.unique(training_labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")

    # Image properties
    print(f"Training image shape: {training_images.shape}")
    print(f"Training labels shape: {training_labels.shape}")
    print(f"Data type of training images: {training_images.dtype}")
    print(f"Range of pixel values: {training_images.min()} - {training_images.max()}")

    # Define the model architecture : Sequential , Activation Function : ReLu
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # Input Layer 
        tf.keras.layers.Dense(128, activation='relu'), #hidden layers # fully connected layers of 128 Neurons
        tf.keras.layers.Dense(10, activation=tf.nn.softmax) #output layer #This layer has 10 neurons, with each neuron corresponding to one of the possible digits (0 through 9) in the MNIST dataset
    ])

    # Compile the model
    model.compile(optimizer='adam', #'adam' is a popular optimizer that combines the benefits of two other extensions of stochastic gradient descent: AdaGrad and RMSProp. The Adam optimizer dynamically adapts the learning rate during training, making it well-suited for a wide range of problems
                  loss='sparse_categorical_crossentropy', #The loss function is a measure of how well the model's predictions match the actual target values during training.
                  metrics=['accuracy']) #performance , Accuracy measures the proportion of correctly classified examples out of all examples.

    # Train the model
    model.fit(training_images, training_labels, epochs=5) #By specifying epochs=5, you're telling the model to go through the entire dataset five times during training. Each epoch consists of multiple iterations

    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

    # Make predictions on the test dataset
    predictions = model.predict(test_images)

    # Loop over the first three test images, display them, and print predictions
    for i in range(3):
        plt.figure(figsize=(2,2))  # Adjust the figure size as needed
        plt.imshow(test_images[i], cmap='gray')
        plt.title(f"Predicted Digit: {np.argmax(predictions[i])}")
        plt.show()
        print(f"Predicted Digit for image {i+1}: {np.argmax(predictions[i])}")

if __name__ == "__main__":
    main()
