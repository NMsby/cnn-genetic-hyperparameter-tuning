"""
Utility functions for genetic algorithm CNN hyperparameter tuning.

This module provides utility functions for displaying and visualizing
results from the genetic algorithm.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any, List
import os

# Create results directory if it doesn't exist
os.makedirs("results/figures", exist_ok=True)


def print_best_individual(individual: Dict[str, Any], fitness: float) -> None:
    """
    Print the hyperparameters and fitness of the best individual.

    Args:
        individual: The individual to print
        fitness: The fitness (validation accuracy) of the individual
    """
    print(f"Best fitness (validation accuracy): {fitness:.4f}")
    print("Best hyperparameters:")
    print(f"  - Number of conv layers: {individual['conv_layers']}")
    print(f"  - Learning rate: {individual['learning_rate']}")

    for i in range(individual['conv_layers']):
        print(f"  - Layer {i + 1}:")
        print(f"    - Filters: {individual[f'filters_{i}']}")
        print(f"    - Kernel size: {individual[f'kernel_size_{i}']}")
        print(f"    - Activation: {individual[f'activation_{i}']}")
        print(f"    - Pool type: {individual[f'pool_type_{i}']}")
        print(f"    - Dropout: {individual[f'dropout_{i}']}")


def plot_fitness_history(fitness_history: List[float], save_path: str = None) -> None:
    """
    Plot the fitness history of the best individual in each generation.

    Args:
        fitness_history: List of best fitness scores from each generation
        save_path: Optional path to save the figure (if None, uses the default path)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.title('Fitness History')
    plt.grid(True)
    plt.xticks(range(len(fitness_history)))

    # Add annotations for improvements
    for i in range(1, len(fitness_history)):
        if fitness_history[i] > fitness_history[i - 1]:
            plt.annotate(f"+{(fitness_history[i] - fitness_history[i - 1]) * 100:.2f}%",
                         (i, fitness_history[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.savefig("results/figures/fitness_history.png", dpi=150, bbox_inches='tight')

    plt.show()
