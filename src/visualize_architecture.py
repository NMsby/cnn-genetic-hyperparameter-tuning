import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import subprocess
import logging
import sys

# Import our model builder
from genetic_algorithms_starter import build_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisualizeArchitecture")


def visualize_architecture(individual, filename="cnn_architecture"):
    """
    Visualize a CNN architecture based on its hyperparameters.

    Args:
        individual: Dictionary containing the hyperparameters
        filename: Base filename for the output image
    """
    logger.info("Visualizing CNN architecture...")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Build the model
    model = build_model(individual)

    # Print model summary to console
    model.summary()

    # Check if graphviz is available for better visualization
    try:
        # Try to create a visualization of the model
        plot_model(
            model,
            to_file=f"results/{filename}.png",
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True
        )
        logger.info(f"Model visualization saved to results/{filename}.png")
    except Exception as e:
        logger.warning(f"Could not create model visualization: {e}")
        logger.warning("To enable visualization, install graphviz and pydot:")
        logger.warning("pip install pydot")
        logger.warning("And install Graphviz from https://graphviz.org/download/")

    # Create a simpler text-based representation for the file
    # Use a custom print function to create a simpler representation without Unicode characters
    layers_info = ["Model: " + model.name, "=" * 80, f"{'Layer (type)':40} {'Output Shape':25} {'Param #':10}",
                   "=" * 80]

    for layer in model.layers:
        output_shape = str(layer.output_shape)
        layers_info.append(
            f"{layer.name + ' (' + layer.__class__.__name__ + ')':40} {output_shape:25} {layer.count_params():10}")

    layers_info.append("=" * 80)
    layers_info.append(f"Total params: {model.count_params():,}")
    layers_info.append(f"Trainable params: {sum(tf.keras.backend.count_params(p) for p in model.trainable_weights):,}")
    layers_info.append(
        f"Non-trainable params: {sum(tf.keras.backend.count_params(p) for p in model.non_trainable_weights):,}")

    # Write to file with UTF-8 encoding
    try:
        with open(f"results/{filename}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(layers_info))
        logger.info(f"Model summary saved to results/{filename}.txt")
    except Exception as e:
        logger.error(f"Error saving model summary to file: {e}")
        # Fallback to ASCII-only version if UTF-8 fails
        try:
            with open(f"results/{filename}_ascii.txt", "w", encoding="ascii", errors="replace") as f:
                f.write("\n".join(layers_info))
            logger.info(f"ASCII model summary saved to results/{filename}_ascii.txt")
        except Exception as e2:
            logger.error(f"Error saving ASCII model summary: {e2}")

    return model


if __name__ == "__main__":
    # Example architecture for visualization
    example_architecture = {
        'conv_layers': 3,
        'learning_rate': 0.001,
        'filters_0': 32,
        'kernel_size_0': 3,
        'activation_0': 'relu',
        'pool_type_0': 'max',
        'dropout_0': 0.25,
        'filters_1': 64,
        'kernel_size_1': 3,
        'activation_1': 'relu',
        'pool_type_1': 'max',
        'dropout_1': 0.25,
        'filters_2': 128,
        'kernel_size_2': 3,
        'activation_2': 'relu',
        'pool_type_2': 'max',
        'dropout_2': 0.5
    }

    # Visualize the example architecture
    model = visualize_architecture(example_architecture, "example_cnn")
