"""
CNN model builder for genetic algorithm hyperparameter tuning.

This module handles loading data, building CNN models, and evaluating models
for the genetic algorithm fitness function.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from typing import Dict, Any, Tuple


def load_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load and preprocess CIFAR-10 dataset.

    Returns:
        Tuple containing three pairs of (data, labels) for training, validation, and test sets:
        - Training set: (x_train, y_train)
        - Validation set: (x_val, y_val)
        - Test set: (x_test, y_test)
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create a validation set
    val_size = 5000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model(hyperparameters: Dict[str, Any]) -> tf.keras.Model:
    """
    Build a CNN model based on provided hyperparameters.

    Args:
        hyperparameters: Dictionary containing hyperparameters for the model

    Returns:
        tf.keras.Model: Compiled CNN model ready for training
    """
    model = Sequential()

    # Add convolutional layers
    for i in range(hyperparameters['conv_layers']):
        # Add convolutional layer
        if i == 0:
            # First layer needs input shape
            model.add(Conv2D(
                filters=hyperparameters[f'filters_{i}'],
                kernel_size=hyperparameters[f'kernel_size_{i}'],
                activation=hyperparameters[f'activation_{i}'],
                padding='same',
                input_shape=(32, 32, 3)  # CIFAR-10 images are 32x32x3
            ))
        else:
            model.add(Conv2D(
                filters=hyperparameters[f'filters_{i}'],
                kernel_size=hyperparameters[f'kernel_size_{i}'],
                activation=hyperparameters[f'activation_{i}'],
                padding='same'
            ))

        # Add pooling layer if specified
        if hyperparameters[f'pool_type_{i}'] == 'max':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif hyperparameters[f'pool_type_{i}'] == 'avg':
            model.add(AveragePooling2D(pool_size=(2, 2)))

        # Add dropout if rate > 0
        if hyperparameters[f'dropout_{i}'] > 0:
            model.add(Dropout(hyperparameters[f'dropout_{i}']))

    # Add flatten layer
    model.add(Flatten())

    # Add dense layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Add output layer
    model.add(Dense(10, activation='softmax'))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def evaluate_fitness(
        individual: Dict[str, Any],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 5,
        batch_size: int = 64
) -> float:
    """
    Build and train a model with the given hyperparameters and return validation accuracy.

    Args:
        individual: The hyperparameters to evaluate
        x_train: Training data features
        y_train: Training data labels
        x_val: Validation data features
        y_val: Validation data labels
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        float: Validation accuracy (fitness score) in range [0, 1]
    """
    # Build model
    model = build_model(individual)

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=0  # Silent training
    )

    # Return validation accuracy as fitness
    return history.history['val_accuracy'][-1]
