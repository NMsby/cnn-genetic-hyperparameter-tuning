import random
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
from pprint import pprint

# Import the necessary functions from our module
from genetic_algorithms_starter import generate_individual, crossover

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def generate_test_parents() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate two test parents for crossover testing.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Two parent individuals
    """
    # Generate parents with controlled properties for better testing
    parent1 = generate_individual()
    parent2 = generate_individual()

    # Ensure parents have different number of layers for testing layer handling
    if parent1['conv_layers'] == parent2['conv_layers']:
        # Modify one parent to have a different number of layers
        if parent1['conv_layers'] < 5:
            parent2['conv_layers'] += 1
        else:
            parent2['conv_layers'] -= 1

        # Ensure the modified parent has all necessary layer parameters
        for i in range(max(parent1['conv_layers'], parent2['conv_layers'])):
            if i < parent2['conv_layers'] and f'filters_{i}' not in parent2:
                parent2[f'filters_{i}'] = random.choice([16, 32, 64, 128, 256])
                parent2[f'kernel_size_{i}'] = random.choice([3, 5, 7])
                parent2[f'activation_{i}'] = random.choice(['relu', 'elu', 'leaky_relu'])
                parent2[f'pool_type_{i}'] = random.choice(['max', 'avg', 'none'])
                parent2[f'dropout_{i}'] = random.choice([0.0, 0.25, 0.5])

    return parent1, parent2


def test_crossover_basic() -> None:
    """Test the basic functionality of the crossover function."""
    print("Testing basic functionality of crossover...")

    # Generate test parents
    parent1, parent2 = generate_test_parents()

    # Print parent properties
    print("\nParent 1:")
    print(f"  Number of layers: {parent1['conv_layers']}")
    print(f"  Learning rate: {parent1['learning_rate']}")
    for i in range(parent1['conv_layers']):
        print(f"  Layer {i}:")
        print(f"    Filters: {parent1[f'filters_{i}']}")
        print(f"    Kernel size: {parent1[f'kernel_size_{i}']}")
        print(f"    Activation: {parent1[f'activation_{i}']}")
        print(f"    Pool type: {parent1[f'pool_type_{i}']}")
        print(f"    Dropout: {parent1[f'dropout_{i}']}")

    print("\nParent 2:")
    print(f"  Number of layers: {parent2['conv_layers']}")
    print(f"  Learning rate: {parent2['learning_rate']}")
    for i in range(parent2['conv_layers']):
        print(f"  Layer {i}:")
        print(f"    Filters: {parent2[f'filters_{i}']}")
        print(f"    Kernel size: {parent2[f'kernel_size_{i}']}")
        print(f"    Activation: {parent2[f'activation_{i}']}")
        print(f"    Pool type: {parent2[f'pool_type_{i}']}")
        print(f"    Dropout: {parent2[f'dropout_{i}']}")

    # Perform crossover
    offspring1, offspring2 = crossover(parent1, parent2)

    # Print offspring properties
    print("\nOffspring 1:")
    print(f"  Number of layers: {offspring1['conv_layers']}")
    print(f"  Learning rate: {offspring1['learning_rate']}")
    for i in range(offspring1['conv_layers']):
        print(f"  Layer {i}:")
        print(f"    Filters: {offspring1[f'filters_{i}']}")
        print(f"    Kernel size: {offspring1[f'kernel_size_{i}']}")
        print(f"    Activation: {offspring1[f'activation_{i}']}")
        print(f"    Pool type: {offspring1[f'pool_type_{i}']}")
        print(f"    Dropout: {offspring1[f'dropout_{i}']}")

    print("\nOffspring 2:")
    print(f"  Number of layers: {offspring2['conv_layers']}")
    print(f"  Learning rate: {offspring2['learning_rate']}")
    for i in range(offspring2['conv_layers']):
        print(f"  Layer {i}:")
        print(f"    Filters: {offspring2[f'filters_{i}']}")
        print(f"    Kernel size: {offspring2[f'kernel_size_{i}']}")
        print(f"    Pool type: {offspring2[f'pool_type_{i}']}")
        print(f"    Activation: {offspring2[f'activation_{i}']}")
        print(f"    Dropout: {offspring2[f'dropout_{i}']}")

    print("\nBasic crossover test completed!")


def test_crossover_validity() -> None:
    """Test if crossover produces valid offspring architectures."""
    print("\nTesting validity of crossover offspring...")

    # Run multiple crossover operations to test validity
    num_tests = 100

    for test in range(num_tests):
        # Generate test parents
        parent1, parent2 = generate_test_parents()

        # Perform crossover
        try:
            offspring1, offspring2 = crossover(parent1, parent2)

            # Verify the basic structure of offspring
            for offspring in [offspring1, offspring2]:
                assert 'conv_layers' in offspring, "Missing conv_layers in offspring"
                assert 'learning_rate' in offspring, "Missing learning_rate in offspring"

                # Verify each layer has all necessary parameters
                for i in range(offspring['conv_layers']):
                    assert f'filters_{i}' in offspring, f"Missing filters_{i} in offspring"
                    assert f'kernel_size_{i}' in offspring, f"Missing kernel_size_{i} in offspring"
                    assert f'activation_{i}' in offspring, f"Missing activation_{i} in offspring"
                    assert f'pool_type_{i}' in offspring, f"Missing pool_type_{i} in offspring"
                    assert f'dropout_{i}' in offspring, f"Missing dropout_{i} in offspring"

            # Verify offspring inherited parameters from parents
            min_layers = min(parent1['conv_layers'], parent2['conv_layers'])
            for i in range(min_layers):
                # Each parameter should come from either parent1 or parent2
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    param_name = f'{param}_{i}'
                    assert (offspring1[param_name] == parent1[param_name] or
                            offspring1[param_name] == parent2[param_name]), \
                        f"Offspring1 {param_name} not inherited from either parent"
                    assert (offspring2[param_name] == parent1[param_name] or
                            offspring2[param_name] == parent2[param_name]), \
                        f"Offspring2 {param_name} not inherited from either parent"

        except Exception as e:
            print(f"Test {test + 1} failed with error: {e}")
            print(f"Parent1: {parent1}")
            print(f"Parent2: {parent2}")
            break
    else:
        print(f"All {num_tests} crossover validity tests passed!")


def analyze_crossover_diversity() -> None:
    """Analyze how crossover affects the diversity of hyperparameters."""
    print("\nAnalyzing parameter inheritance in crossover...")

    # Track inheritance statistics
    num_tests = 1000
    inheritance_counts = {
        'learning_rate': {'inherited_p1': 0, 'inherited_p2': 0},
        'param_swaps': 0,
        'total_params': 0
    }

    for _ in range(num_tests):
        # Generate test parents
        parent1, parent2 = generate_test_parents()

        # Perform crossover
        offspring1, offspring2 = crossover(parent1, parent2)

        # Check learning rate inheritance
        if offspring1['learning_rate'] == parent1['learning_rate']:
            inheritance_counts['learning_rate']['inherited_p1'] += 1
        elif offspring1['learning_rate'] == parent2['learning_rate']:
            inheritance_counts['learning_rate']['inherited_p2'] += 1

        # Check layer parameter swaps
        min_layers = min(parent1['conv_layers'], parent2['conv_layers'])
        for i in range(min_layers):
            for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                param_name = f'{param}_{i}'
                inheritance_counts['total_params'] += 1

                # Count when offspring1 gets param from parent2 (indicating a swap)
                if offspring1[param_name] == parent2[param_name]:
                    inheritance_counts['param_swaps'] += 1

    # Print inheritance statistics
    print("\nLearning Rate Inheritance:")
    p1_pct = inheritance_counts['learning_rate']['inherited_p1'] / num_tests * 100
    p2_pct = inheritance_counts['learning_rate']['inherited_p2'] / num_tests * 100
    print(f"  From Parent 1: {p1_pct:.1f}%")
    print(f"  From Parent 2: {p2_pct:.1f}%")

    swap_pct = inheritance_counts['param_swaps'] / inheritance_counts['total_params'] * 100
    print(f"\nLayer Parameter Swap Rate: {swap_pct:.1f}%")
    print(f"  This should be close to 50% for random mixing")

    print("\nCrossover diversity analysis completed!")


if __name__ == "__main__":
    # Run all tests
    test_crossover_basic()
    test_crossover_validity()
    analyze_crossover_diversity()
