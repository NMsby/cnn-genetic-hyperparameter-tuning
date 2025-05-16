import random
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
from pprint import pprint

# Import functions from our module
from genetic_algorithms_starter import generate_individual, select_parents

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def generate_test_population(population_size: int = 10) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Generate a test population with random fitness scores.

    Args:
        population_size (int): Size of the population to generate

    Returns:
        Tuple[List[Dict[str, Any]], List[float]]: Population and fitness scores
    """
    # Generate random population
    population = [generate_individual() for _ in range(population_size)]

    # Assign random fitness scores between 0 and 1
    fitness_scores = [random.random() for _ in range(population_size)]

    return population, fitness_scores


def test_select_parents_basic() -> None:
    """Test the basic functionality of the select_parents function."""
    print("Testing basic functionality of select_parents...")

    # Generate test population and fitness scores
    population_size = 10
    population, fitness_scores = generate_test_population(population_size)

    # Print population with fitness scores
    print("\nTest Population:")
    for i, (individual, fitness) in enumerate(zip(population, fitness_scores)):
        print(f"Individual {i}: Fitness = {fitness:.4f}, Layers = {individual['conv_layers']}")

    # Select parents
    num_parents = 5
    parents = select_parents(population, fitness_scores, num_parents)

    # Print selected parents
    print("\nSelected Parents:")
    for i, parent in enumerate(parents):
        # Find the parent's index in the original population
        parent_idx = next(i for i, ind in enumerate(population) if ind is parent)
        print(f"Parent {i}: Original Index = {parent_idx}, Fitness = {fitness_scores[parent_idx]:.4f}")

    # Verify the correct number of parents
    assert len(parents) == num_parents, f"Expected {num_parents} parents, got {len(parents)}"

    print("\nBasic test passed!")


def test_selection_pressure() -> None:
    """Test if tournament selection favors individuals with higher fitness."""
    print("\nTesting selection pressure...")

    # Create a population with controlled fitness scores
    population_size = 20
    population = [generate_individual() for _ in range(population_size)]

    # Assign linearly increasing fitness scores
    fitness_scores = [i / population_size for i in range(population_size)]

    print("\nTest Population with Linear Fitness:")
    for i, fitness in enumerate(fitness_scores):
        print(f"Individual {i}: Fitness = {fitness:.4f}")

    # Run multiple selection trials
    num_trials = 1000
    selection_counts = [0] * population_size

    for _ in range(num_trials):
        # Select one parent
        parent = select_parents(population, fitness_scores, 1)[0]
        # Find and count the selected individual
        parent_idx = next(i for i, ind in enumerate(population) if ind is parent)
        selection_counts[parent_idx] += 1

    # Print selection results
    print("\nSelection Results after", num_trials, "trials:")
    for i, count in enumerate(selection_counts):
        print(
            f"Individual {i}: Fitness = {fitness_scores[i]:.4f}, Selected {count} times ({count / num_trials * 100:.2f}%)")

    # Check if higher fitness individuals were selected more frequently
    # Split the population into quartiles and verify selection frequency increases
    quartile_size = population_size // 4
    quartile_counts = [
        sum(selection_counts[:quartile_size]),
        sum(selection_counts[quartile_size:2 * quartile_size]),
        sum(selection_counts[2 * quartile_size:3 * quartile_size]),
        sum(selection_counts[3 * quartile_size:])
    ]

    print("\nQuartile Selection Frequencies:")
    for i, count in enumerate(quartile_counts):
        print(f"Quartile {i + 1}: Selected {count} times ({count / num_trials * 100:.2f}%)")

    # Verify selection pressure (higher quartiles should be selected more often)
    for i in range(3):
        assert quartile_counts[i] < quartile_counts[i + 1], \
            f"Selection pressure test failed: Quartile {i + 1} was selected more than Quartile {i + 2}"

    print("\nSelection pressure test passed!")


def test_tournament_size_effect() -> None:
    """Test how different tournament sizes affect selection pressure."""
    print("\nTesting effect of tournament size...")

    # Create a population with controlled fitness scores
    population_size = 100
    population = [generate_individual() for _ in range(population_size)]

    # Assign linearly increasing fitness scores
    fitness_scores = [i / population_size for i in range(population_size)]

    # Test different tournament sizes
    tournament_sizes = [1, 2, 3, 5, 10]
    num_trials = 1000
    num_parents = 10

    print("\nAverage fitness of selected parents with different tournament sizes:")

    for tournament_size in tournament_sizes:
        # Run multiple selection trials
        total_fitness = 0

        for _ in range(num_trials):
            parents = select_parents(population, fitness_scores, num_parents, tournament_size)
            parent_indices = [next(i for i, ind in enumerate(population) if ind is parent) for parent in parents]
            selected_fitness = [fitness_scores[idx] for idx in parent_indices]
            total_fitness += sum(selected_fitness) / len(selected_fitness)

        avg_fitness = total_fitness / num_trials
        print(f"Tournament Size {tournament_size}: Average Fitness = {avg_fitness:.4f}")

    print("\nTournament size test completed!")


if __name__ == "__main__":
    # Run all tests
    test_select_parents_basic()
    test_selection_pressure()
    test_tournament_size_effect()
