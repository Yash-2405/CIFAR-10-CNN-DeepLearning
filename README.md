# CIFAR-10 Image Classification Using Convolutional Neural Networks (CNNs)

# Introduction

This project demonstrates the application of Convolutional Neural Networks (CNNs) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

# Dataset Description

The CIFAR-10 dataset contains the following classes:

Airplanes

Cars

Birds

Cats

Deer

Dogs

Frogs

Horses

Ships

Trucks

Each class has 6,000 images, split into 50,000 training images and 10,000 testing images.

# Data Preprocessing

Loading the Dataset: The dataset is loaded using TensorFlow's keras.datasets module.

Normalization: Pixel values are normalized to the range [0, 1] to facilitate faster and more efficient training.

Data Augmentation: Techniques such as rotation, flipping, and zooming are applied to the training images to increase the diversity of the dataset and improve model generalization.

# Model Architecture

The CNN architecture is built using TensorFlow and Keras, consisting of:

Convolutional layers with ReLU activation functions

Max-pooling layers

Dropout layers to prevent overfitting

Dense layers with softmax activation for the output layer

Training the Model

Loss Function: Categorical cross-entropy

Optimizer: Adam

Epochs: 50

Batch Size: 64

# Model Evaluation

The model's performance is evaluated using accuracy and loss metrics on the test dataset. The final trained model achieves high accuracy, demonstrating the effectiveness of CNNs in image classification tasks.

# Sample Images

Here are some sample images from the CIFAR-10 dataset used in this project:

![image](https://github.com/Yash-2405/CIFAR-10-CNN-DeepLearning/assets/115539756/60fa94ab-6390-404b-a3ec-f8341785b166)

# Conclusion

This project illustrates the power of Convolutional Neural Networks (CNNs) in performing image classification tasks. Experimenting with various architectures, hyperparameters, and data augmentation methods can lead to further advancements in model performance.
