import random
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
from pprint import pprint

# Import the necessary functions from our module
from genetic_algorithms_starter import (
    generate_individual, select_parents, crossover, mutate,
    evolve_population, HYPERPARAMETER_SPACE
)

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


def test_evolve_population_basic() -> None:
    """Test the basic functionality of the evolve_population function."""
    print("Testing basic functionality of evolve_population...")

    # Generate test population and fitness scores
    population_size = 10
    population, fitness_scores = generate_test_population(population_size)

    # Print population with fitness scores
    print("\nInitial Population:")
    for i, (individual, fitness) in enumerate(zip(population, fitness_scores)):
        print(f"Individual {i}: Fitness = {fitness:.4f}, Layers = {individual['conv_layers']}")

    # Evolve population
    next_generation = evolve_population(population, fitness_scores, population_size)

    # Print new generation
    print("\nNext Generation:")
    for i, individual in enumerate(next_generation):
        print(f"Individual {i}: Layers = {individual['conv_layers']}")

    # Verify the correct population size
    assert len(next_generation) == population_size, \
        f"Expected population size {population_size}, got {len(next_generation)}"

    print("\nBasic evolution test passed!")


def test_elitism() -> None:
    """Test if the best individuals are preserved through elitism."""
    print("\nTesting elitism...")

    # Generate test population and fitness scores
    population_size = 10
    population, fitness_scores = generate_test_population(population_size)

    # Manually set some high fitness scores to identify elites clearly
    best_index = random.randint(0, population_size - 1)
    second_best_index = random.randint(0, population_size - 1)
    while second_best_index == best_index:
        second_best_index = random.randint(0, population_size - 1)

    fitness_scores[best_index] = 0.99  # Very high fitness
    fitness_scores[second_best_index] = 0.95  # Second highest

    # Print elites
    print(f"Best individual (index {best_index}): Fitness = {fitness_scores[best_index]:.4f}")
    print(f"Second best individual (index {second_best_index}): Fitness = {fitness_scores[second_best_index]:.4f}")

    # Store copies of the elites to check if they're preserved
    best_individual = population[best_index].copy()
    second_best_individual = population[second_best_index].copy()

    # Evolve population
    next_generation = evolve_population(population, fitness_scores, population_size)

    # Check if the best individual was preserved (elitism)
    best_preserved = False
    for individual in next_generation:
        # Check if this individual is the same as the best individual
        if all(
                individual[key] == best_individual[key]
                for key in best_individual
                if key in individual
        ) and all(
            individual[key] == best_individual[key]
            for key in individual
            if key in best_individual
        ):
            best_preserved = True
            break

    assert best_preserved, "Best individual was not preserved (elitism failed)"

    print("Elitism test passed!")


def test_evolution_multiple_generations() -> None:
    """Test evolution over multiple generations to check for fitness improvement."""
    print("\nTesting multiple generations of evolution...")

    # Define evolution parameters
    population_size = 20
    num_generations = 5
    mutation_rate = 0.1

    # Generate initial population with controlled fitness values
    # We'll make fitness proportional to the number of layers
    population = []
    fitness_scores = []

    for _ in range(population_size):
        individual = generate_individual()

        # Assign fitness based on the number of layers (just for testing)
        # This creates a bias toward more layers in our simple test
        fitness = individual['conv_layers'] / 5.0  # Normalize to [0-1]

        population.append(individual)
        fitness_scores.append(fitness)

    # Track average fitness and layer counts across generations
    generation_stats = []

    # Store initial statistics
    avg_fitness = sum(fitness_scores) / len(fitness_scores)
    avg_layers = sum(ind['conv_layers'] for ind in population) / len(population)
    generation_stats.append((avg_fitness, avg_layers))

    print(f"Initial Population: Avg Fitness = {avg_fitness:.4f}, Avg Layers = {avg_layers:.2f}")

    # Evolve for multiple generations
    current_population = population
    current_fitness = fitness_scores

    for gen in range(num_generations):
        # Evolve to next generation
        next_generation = evolve_population(
            current_population, current_fitness, population_size, mutation_rate
        )

        # Calculate new fitness scores (based on our simple layer-count model)
        next_fitness = [ind['conv_layers'] / 5.0 for ind in next_generation]

        # Calculate statistics
        avg_fitness = sum(next_fitness) / len(next_fitness)
        avg_layers = sum(ind['conv_layers'] for ind in next_generation) / len(next_generation)
        generation_stats.append((avg_fitness, avg_layers))

        print(f"Generation {gen + 1}: Avg Fitness = {avg_fitness:.4f}, Avg Layers = {avg_layers:.2f}")

        # Update current population for next iteration
        current_population = next_generation
        current_fitness = next_fitness

    # Check for fitness improvement over generations
    initial_fitness = generation_stats[0][0]
    final_fitness = generation_stats[-1][0]

    print(f"\nFitness improvement: {initial_fitness:.4f} -> {final_fitness:.4f}")

    # In our simple test, fitness should generally increase,
    # But we won't assert this hard condition because genetic algorithms have
    # randomness that might not always show improvement in just a few generations
    if final_fitness > initial_fitness:
        print("Fitness improved as expected!")
    else:
        print("Fitness did not improve in this run (could be due to randomness).")

    print("\nMultiple generations test completed!")


def test_population_validity() -> None:
    """Test if all individuals in the evolved population are valid."""
    print("\nTesting validity of evolved population...")

    # Generate test population and fitness scores
    population_size = 15
    population, fitness_scores = generate_test_population(population_size)

    # Evolve population
    next_generation = evolve_population(population, fitness_scores, population_size)

    # Verify each individual is valid
    for i, individual in enumerate(next_generation):
        # Check essential hyperparameters
        assert 'conv_layers' in individual, f"Individual {i} missing 'conv_layers'"
        assert 'learning_rate' in individual, f"Individual {i} missing 'learning_rate'"

        # Check values are in valid ranges
        assert individual['conv_layers'] in HYPERPARAMETER_SPACE['conv_layers'], \
            f"Individual {i} has invalid 'conv_layers': {individual['conv_layers']}"
        assert individual['learning_rate'] in HYPERPARAMETER_SPACE['learning_rates'], \
            f"Individual {i} has invalid 'learning_rate': {individual['learning_rate']}"

        # Check layer parameters
        for j in range(individual['conv_layers']):
            # Check all layer parameters exist
            for param, space_key in [
                ('filters', 'filters'),
                ('kernel_size', 'kernel_sizes'),
                ('activation', 'activation_functions'),
                ('pool_type', 'pool_types'),
                ('dropout', 'dropout_rates')
            ]:
                param_name = f'{param}_{j}'

                assert param_name in individual, \
                    f"Individual {i} missing '{param_name}'"
                assert individual[param_name] in HYPERPARAMETER_SPACE[space_key], \
                    f"Individual {i} has invalid '{param_name}': {individual[param_name]}"

    print("All individuals in evolved population are valid!")


if __name__ == "__main__":
    # Run all tests
    test_evolve_population_basic()
    test_elitism()
    test_evolution_multiple_generations()
    test_population_validity()
