import random
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
from pprint import pprint
from copy import deepcopy

# Import the necessary functions from our module
from genetic_algorithms_starter import generate_individual, mutate, HYPERPARAMETER_SPACE

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def test_mutate_basic() -> None:
    """Test the basic functionality of the mutate function."""
    print("Testing basic functionality of mutate...")

    # Generate a test individual
    original = generate_individual()

    # Print original properties
    print("\nOriginal Individual:")
    print(f"  Number of layers: {original['conv_layers']}")
    print(f"  Learning rate: {original['learning_rate']}")
    for i in range(original['conv_layers']):
        print(f"  Layer {i}:")
        print(f"    Filters: {original[f'filters_{i}']}")
        print(f"    Kernel size: {original[f'kernel_size_{i}']}")
        print(f"    Activation: {original[f'activation_{i}']}")
        print(f"    Pool type: {original[f'pool_type_{i}']}")
        print(f"    Dropout: {original[f'dropout_{i}']}")

    # Perform mutation with a high rate to see changes
    mutated = mutate(original, mutation_rate=0.5)

    # Print mutated properties
    print("\nMutated Individual (50% mutation rate):")
    print(f"  Number of layers: {mutated['conv_layers']}")
    print(f"  Learning rate: {mutated['learning_rate']}")
    for i in range(mutated['conv_layers']):
        print(f"  Layer {i}:")
        print(f"    Filters: {mutated[f'filters_{i}']}")
        print(f"    Kernel size: {mutated[f'kernel_size_{i}']}")
        print(f"    Activation: {mutated[f'activation_{i}']}")
        print(f"    Pool type: {mutated[f'pool_type_{i}']}")
        print(f"    Dropout: {mutated[f'dropout_{i}']}")

    # Verify that the original was not modified
    print("\nVerifying original individual was not modified...")
    assert id(original) != id(mutated), "Original individual was modified (same object reference)"
    if original['conv_layers'] == mutated['conv_layers']:
        for i in range(original['conv_layers']):
            for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                param_name = f'{param}_{i}'
                if original[param_name] != mutated[param_name]:
                    print(f"  Parameter {param_name} was mutated: {original[param_name]} -> {mutated[param_name]}")

    print("\nBasic mutation test completed!")


def test_mutation_rate_effect() -> None:
    """Test how different mutation rates affect the degree of change."""
    print("\nTesting effect of different mutation rates...")

    # Generate test rates
    mutation_rates = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    num_tests = 100

    for rate in mutation_rates:
        # Count changes across multiple mutations
        param_changes = 0
        total_params = 0
        layer_changes = 0

        for _ in range(num_tests):
            # Generate a random individual
            original = generate_individual()
            original_copy = deepcopy(original)

            # Mutate the individual
            mutated = mutate(original, mutation_rate=rate)

            # Check if the number of layers changed
            if original['conv_layers'] != mutated['conv_layers']:
                layer_changes += 1

            # Count parameter changes
            min_layers = min(original['conv_layers'], mutated['conv_layers'])

            # Check learning rate
            total_params += 1
            if original['learning_rate'] != mutated['learning_rate']:
                param_changes += 1

            # Check layer parameters
            for i in range(min_layers):
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    param_name = f'{param}_{i}'
                    total_params += 1
                    if original[param_name] != mutated[param_name]:
                        param_changes += 1

            # Verify original wasn't changed
            assert original == original_copy, "Original individual was modified"

        # Calculate change percentages
        param_change_pct = param_changes / total_params * 100
        layer_change_pct = layer_changes / num_tests * 100

        print(f"Mutation rate {rate}:")
        print(f"  Parameter change percentage: {param_change_pct:.2f}%")
        print(f"  Layer count change percentage: {layer_change_pct:.2f}%")

    print("\nMutation rate effect test completed!")


def test_mutate_validity() -> None:
    """Test if mutation produces valid architectures."""
    print("\nTesting validity of mutated individuals...")

    # Run multiple mutation operations to test validity
    num_tests = 100
    high_mutation_rate = 0.8  # High rate to trigger more changes

    for test in range(num_tests):
        # Generate a random individual
        original = generate_individual()

        # Mutate with high rate to ensure changes
        try:
            mutated = mutate(original, mutation_rate=high_mutation_rate)

            # Verify basic structure
            assert 'conv_layers' in mutated, "Missing conv_layers in mutated individual"
            assert 'learning_rate' in mutated, "Missing learning_rate in mutated individual"
            assert mutated['conv_layers'] in HYPERPARAMETER_SPACE['conv_layers'], \
                f"Invalid conv_layers value: {mutated['conv_layers']}"
            assert mutated['learning_rate'] in HYPERPARAMETER_SPACE['learning_rates'], \
                f"Invalid learning_rate value: {mutated['learning_rate']}"

            # Verify each layer has all necessary parameters with valid values
            for i in range(mutated['conv_layers']):
                for param, space_key in [
                    ('filters', 'filters'),
                    ('kernel_size', 'kernel_sizes'),
                    ('activation', 'activation_functions'),
                    ('pool_type', 'pool_types'),
                    ('dropout', 'dropout_rates')
                ]:
                    param_name = f'{param}_{i}'
                    assert param_name in mutated, f"Missing {param_name} in mutated individual"
                    assert mutated[param_name] in HYPERPARAMETER_SPACE[space_key], \
                        f"Invalid {param_name} value: {mutated[param_name]}"

            # Verify no extra layer parameters exist beyond the number of layers
            for i in range(mutated['conv_layers'], 10):  # Assuming max layers is less than 10
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    param_name = f'{param}_{i}'
                    assert param_name not in mutated, \
                        f"Extra parameter {param_name} found in mutated individual"

        except Exception as e:
            print(f"Test {test + 1} failed with error: {e}")
            print(f"Original: {original}")
            print(f"Mutated: {mutated}")
            break
    else:
        print(f"All {num_tests} mutation validity tests passed!")


def test_layer_changes() -> None:
    """Test specific behavior when adding or removing layers during mutation."""
    print("\nTesting layer addition and removal during mutation...")

    # Test adding layers
    print("\nTesting layer addition:")
    # Create an individual with minimal layers
    individual = generate_individual()
    individual['conv_layers'] = 1  # Force to have only 1 layer

    # Simulate mutation that increases layers (by setting random seed if needed)
    # Here we'll directly modify the number of layers for testing
    mutated = individual.copy()
    mutated['conv_layers'] = 3  # Increase to 3 layers

    # Now call our mutate function on this "pre-modified" individual
    # with a mutation rate that won't change the conv_layers again
    mutated = mutate(mutated, mutation_rate=0.0)

    # Check that all necessary parameters were added
    for i in range(1, 3):  # Check layers 1 and 2 (added layers)
        for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
            param_name = f'{param}_{i}'
            assert param_name in mutated, f"Parameter {param_name} not added when increasing layers"

    # Test removing layers
    print("\nTesting layer removal:")
    # Create an individual with maximum layers
    individual = generate_individual()
    individual['conv_layers'] = 5  # Force to have 5 layers

    # Ensure it has parameters for all 5 layers
    for i in range(5):
        for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
            param_name = f'{param}_{i}'
            if param_name not in individual:
                individual[param_name] = random.choice(HYPERPARAMETER_SPACE[param.rstrip('_0123456789') + 's'])

    # Simulate mutation that decreases layers
    mutated = individual.copy()
    mutated['conv_layers'] = 2  # Decrease to 2 layers

    # Call our mutate function with a rate that won't change conv_layers again
    mutated = mutate(mutated, mutation_rate=0.0)

    # Check that excess parameters were removed
    for i in range(2, 5):  # Check layers 2, 3, and 4 (removed layers)
        for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
            param_name = f'{param}_{i}'
            assert param_name not in mutated, f"Parameter {param_name} not removed when decreasing layers"

    print("Layer addition and removal tests passed!")


if __name__ == "__main__":
    # Run all tests
    test_mutate_basic()
    test_mutation_rate_effect()
    test_mutate_validity()
    test_layer_changes()
