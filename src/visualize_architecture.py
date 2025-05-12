import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import subprocess
import logging

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

    # Print model summary
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

    # Create a text-based representation as backup
    summary_io = StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
    with open(f"results/{filename}.txt", "w") as f:
        f.write(summary_io.getvalue())
    logger.info(f"Model summary saved to results/{filename}.txt")

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
