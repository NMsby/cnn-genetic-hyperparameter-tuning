import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import subprocess
import logging
import sys
from contextlib import redirect_stdout

# Import our model builder
from genetic_algorithms_starter import build_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisualizeArchitecture")


def create_architecture_diagram(model, filename="cnn_architecture_diagram"):
    """
    Create a simple diagram of model architecture using matplotlib.
    This is an alternative when Graphviz is not available.

    Args:
        model: The Keras model
        filename: Base filename for the output image
    """
    # Create a figure
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()

    # Define colors for different layer types
    layer_colors = {
        'Conv2D': '#FFA07A',  # Light Salmon
        'MaxPooling2D': '#20B2AA',  # Light Sea Green
        'AveragePooling2D': '#87CEFA',  # Light Sky Blue
        'Dropout': '#FAFAD2',  # Light Goldenrod Yellow
        'Flatten': '#9370DB',  # Medium Purple
        'Dense': '#90EE90',  # Light Green
    }

    # Default color for other layer types
    default_color = '#D3D3D3'  # Light Gray

    # Define box dimensions
    box_width = 0.8
    box_height = 0.5
    y_spacing = 1.2

    # Track total layers for vertical positioning
    total_layers = len(model.layers)

    # Draw each layer
    for i, layer in enumerate(model.layers):
        # Get layer type and color
        layer_type = layer.__class__.__name__
        color = layer_colors.get(layer_type, default_color)

        # Calculate position (centered horizontally, stacked vertically from top to bottom)
        y_pos = total_layers - i - 1  # Reverse order for top-to-bottom flow

        # Draw the layer box
        rect = plt.Rectangle((0.5 - box_width / 2, y_pos * y_spacing),
                             box_width, box_height,
                             facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)

        # Add layer name and type
        plt.text(0.5, y_pos * y_spacing + box_height / 2,
                 f"{layer.name}\n({layer_type})",
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=10)

        # Add parameter count
        params = layer.count_params()
        if params > 0:
            plt.text(0.5 + box_width / 2 + 0.05, y_pos * y_spacing + box_height / 2,
                     f"Params: {params:,}",
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=8)

        # Draw arrows between layers
        if i < total_layers - 1:
            plt.arrow(0.5, y_pos * y_spacing, 0, -0.7,
                      head_width=0.05, head_length=0.1, fc='black', ec='black')

    # Set plot limits
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, total_layers * y_spacing)

    # Remove axes
    ax.axis('off')

    # Add title
    plt.title(f'Model Architecture: {model.name}')

    # Add total parameters
    plt.figtext(0.5, 0.01, f'Total Parameters: {model.count_params():,}',
                horizontalalignment='center', fontsize=10)

    # Save figure
    plt.savefig(f"results/{filename}.png", dpi=150, bbox_inches='tight')
    logger.info(f"Architecture diagram saved to results/{filename}.png")

    # Close the figure to free memory
    plt.close(fig)


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

    # Try to use plot_model (requires Graphviz)
    graphviz_works = False
    try:
        # Try to create a visualization of the model
        plot_model(
            model,
            to_file=f"results/{filename}.png",
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True
        )
        logger.info(f"Graphviz model visualization saved to results/{filename}.png")
        graphviz_works = True
    except Exception as e:
        logger.warning(f"Could not create Graphviz visualization: {e}")
        logger.warning("To enable Graphviz visualization, install graphviz and pydot:")
        logger.warning("pip install pydot")
        logger.warning("And install Graphviz from https://graphviz.org/download/")

    # Create our custom matplotlib visualization (as a fallback or additional visualization)
    if not graphviz_works:
        create_architecture_diagram(model, filename)

    # Save model summary to a text file
    try:
        # Redirect model.summary() to a file with utf-8 encoding
        with open(f"results/{filename}.txt", "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                model.summary(line_length=80, positions=[.33, .65, .8, 1.])

        logger.info(f"Model summary saved to results/{filename}.txt")
    except Exception as e:
        logger.error(f"Error saving model summary to file: {e}")

        # Alternative approach - create a very simple summary
        try:
            with open(f"results/{filename}_simple.txt", "w", encoding="ascii", errors="replace") as f:
                f.write(f"Model: {model.name}\n\n")
                f.write("Layer Information:\n")
                f.write("-" * 80 + "\n")

                for i, layer in enumerate(model.layers):
                    f.write(f"Layer {i}: {layer.name} ({layer.__class__.__name__})\n")
                    f.write(f"  Parameters: {layer.count_params():,}\n")
                    f.write("\n")

                f.write("-" * 80 + "\n")
                f.write(f"Total parameters: {model.count_params():,}\n")

            logger.info(f"Simple model summary saved to results/{filename}_simple.txt")
        except Exception as e2:
            logger.error(f"Error saving simple model summary: {e2}")

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
