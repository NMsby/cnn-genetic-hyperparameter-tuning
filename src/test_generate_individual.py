# src/test_generate_individual.py

import random
import numpy as np
import tensorflow as tf
from pprint import pprint
from typing import Dict, Any, List

# Import our modified function
from genetic_algorithms_starter import generate_individual, HYPERPARAMETER_SPACE

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def test_generate_individual() -> None:
    """Test the generate_individual function."""
    print("Testing generate_individual function...")

    # Generate 5 individuals for testing
    individuals = [generate_individual() for _ in range(5)]

    for i, individual in enumerate(individuals):
        print(f"\nIndividual {i + 1}:")
        pprint(individual)

        # Validate the individual has all required hyperparameters
        validate_individual(individual)

    print("\nAll individuals are valid!")


def validate_individual(individual: Dict[str, Any]) -> None:
    """
    Validate that an individual has all required hyperparameters and they are valid.

    Args:
        individual: The individual to validate

    Raises:
        AssertionError: If the individual is invalid
    """
    # Check required top-level hyperparameters
    assert 'conv_layers' in individual, "Missing 'conv_layers'"
    assert 'learning_rate' in individual, "Missing 'learning_rate'"

    # Check values are from the search space
    assert individual['conv_layers'] in HYPERPARAMETER_SPACE['conv_layers'], \
        f"Invalid conv_layers value: {individual['conv_layers']}"
    assert individual['learning_rate'] in HYPERPARAMETER_SPACE['learning_rates'], \
        f"Invalid learning_rate value: {individual['learning_rate']}"

    # Check layer-specific hyperparameters
    for i in range(individual['conv_layers']):
        # Check each layer has all required hyperparameters
        assert f'filters_{i}' in individual, f"Missing 'filters_{i}'"
        assert f'kernel_size_{i}' in individual, f"Missing 'kernel_size_{i}'"
        assert f'activation_{i}' in individual, f"Missing 'activation_{i}'"
        assert f'pool_type_{i}' in individual, f"Missing 'pool_type_{i}'"
        assert f'dropout_{i}' in individual, f"Missing 'dropout_{i}'"

        # Check values are from the search space
        assert individual[f'filters_{i}'] in HYPERPARAMETER_SPACE['filters'], \
            f"Invalid filters_{i} value: {individual[f'filters_{i}']}"
        assert individual[f'kernel_size_{i}'] in HYPERPARAMETER_SPACE['kernel_sizes'], \
            f"Invalid kernel_size_{i} value: {individual[f'kernel_size_{i}']}"
        assert individual[f'activation_{i}'] in HYPERPARAMETER_SPACE['activation_functions'], \
            f"Invalid activation_{i} value: {individual[f'activation_{i}']}"
        assert individual[f'pool_type_{i}'] in HYPERPARAMETER_SPACE['pool_types'], \
            f"Invalid pool_type_{i} value: {individual[f'pool_type_{i}']}"
        assert individual[f'dropout_{i}'] in HYPERPARAMETER_SPACE['dropout_rates'], \
            f"Invalid dropout_{i} value: {individual[f'dropout_{i}']}"


def analyze_population_diversity(population_size: int = 100) -> None:
    """
    Generate multiple individuals and analyze the distribution of hyperparameters.

    Args:
        population_size: Number of individuals to generate
    """
    print(f"\nAnalyzing diversity in a population of {population_size} individuals...")

    # Generate a population
    population = [generate_individual() for _ in range(population_size)]

    # Analyze distribution of top-level hyperparameters
    conv_layers_counts = {}
    learning_rate_counts = {}

    for individual in population:
        # Count conv_layers
        conv_layers = individual['conv_layers']
        conv_layers_counts[conv_layers] = conv_layers_counts.get(conv_layers, 0) + 1

        # Count learning_rates
        lr = individual['learning_rate']
        learning_rate_counts[lr] = learning_rate_counts.get(lr, 0) + 1

    # Print analysis
    print("\nConvolutional Layers Distribution:")
    for layers, count in sorted(conv_layers_counts.items()):
        print(f"  {layers} layers: {count} individuals ({count / population_size * 100:.1f}%)")

    print("\nLearning Rate Distribution:")
    for lr, count in sorted(learning_rate_counts.items()):
        print(f"  {lr}: {count} individuals ({count / population_size * 100:.1f}%)")

    # Analyze layer-specific hyperparameters for the first layer (layer 0)
    filters_counts = {}
    kernel_size_counts = {}
    activation_counts = {}
    pool_type_counts = {}
    dropout_counts = {}

    for individual in population:
        if individual['conv_layers'] > 0:  # Only if the individual has at least one layer
            # Count filters
            filters = individual['filters_0']
            filters_counts[filters] = filters_counts.get(filters, 0) + 1

            # Count kernel sizes
            kernel_size = individual['kernel_size_0']
            kernel_size_counts[kernel_size] = kernel_size_counts.get(kernel_size, 0) + 1

            # Count activations
            activation = individual['activation_0']
            activation_counts[activation] = activation_counts.get(activation, 0) + 1

            # Count pool types
            pool_type = individual['pool_type_0']
            pool_type_counts[pool_type] = pool_type_counts.get(pool_type, 0) + 1

            # Count dropout rates
            dropout = individual['dropout_0']
            dropout_counts[dropout] = dropout_counts.get(dropout, 0) + 1

    print("\nLayer 0 Hyperparameter Distributions:")

    print("\n  Filters Distribution:")
    for filters, count in sorted(filters_counts.items()):
        print(f"    {filters}: {count} individuals ({count / population_size * 100:.1f}%)")

    print("\n  Kernel Size Distribution:")
    for kernel_size, count in sorted(kernel_size_counts.items()):
        print(f"    {kernel_size}: {count} individuals ({count / population_size * 100:.1f}%)")

    print("\n  Activation Function Distribution:")
    for activation, count in sorted(activation_counts.items()):
        print(f"    {activation}: {count} individuals ({count / population_size * 100:.1f}%)")

    print("\n  Pool Type Distribution:")
    for pool_type, count in sorted(pool_type_counts.items()):
        print(f"    {pool_type}: {count} individuals ({count / population_size * 100:.1f}%)")

    print("\n  Dropout Rate Distribution:")
    for dropout, count in sorted(dropout_counts.items()):
        print(f"    {dropout}: {count} individuals ({count / population_size * 100:.1f}%)")


if __name__ == "__main__":
    # Test the generate_individual function
    test_generate_individual()

    # Analyze diversity in a larger population
    analyze_population_diversity(population_size=100)
