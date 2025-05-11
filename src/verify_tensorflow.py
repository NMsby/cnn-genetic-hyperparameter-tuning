import tensorflow as tf
import numpy as np
import time

print("TensorFlow version:", tf.__version__)
print("Checking GPU availability...")
print("GPU Available:", tf.config.list_physical_devices('GPU'))

print("\nTesting CIFAR-10 data loading...")
start_time = time.time()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
end_time = time.time()

print(f"Data loaded successfully in {end_time - start_time:.2f} seconds.")
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Test simple model creation
print("\nTesting simple model creation...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model created successfully.")
model.summary()

print("\nEnvironment verification completed successfully.")
