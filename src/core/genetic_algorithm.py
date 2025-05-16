"""
Genetic algorithm implementation for CNN hyperparameter tuning.

This module contains the core genetic algorithm components for optimizing
CNN hyperparameters on image classification tasks.
"""

import numpy as np
import random
import time
from typing import List, Dict, Tuple, Any, Optional
from .model_builder import load_data, build_model, evaluate_fitness

# Define hyperparameter search space
# These are the parameters the genetic algorithm will optimize
HYPERPARAMETER_SPACE = {
    'conv_layers': [1, 2, 3, 4, 5],  # Number of convolutional layers
    'filters': [16, 32, 64, 128, 256],  # Number of filters per layer
    'kernel_sizes': [3, 5, 7],  # Kernel sizes for convolutional layers
    'pool_types': ['max', 'avg', 'none'],  # Pooling types
    'learning_rates': [0.1, 0.01, 0.001, 0.0001],  # Learning rates
    'activation_functions': ['relu', 'elu', 'leaky_relu'],  # Activation functions
    'dropout_rates': [0.0, 0.25, 0.5]  # Dropout rates
}


def generate_individual() -> Dict[str, Any]:
    """
    Generate a random individual (set of hyperparameters).

    An individual represents a specific CNN architecture configuration.
    This function randomly selects values for all hyperparameters from the
    defined search space (HYPERPARAMETER_SPACE).

    Returns:
        Dict[str, Any]: Dictionary containing randomly selected hyperparameters
    """
    # Create a dictionary with all necessary hyperparameters
    # Example Structure:

    # Start with basic hyperparameters
    individual = {
        # Randomly select the number of convolutional layers
        'conv_layers': random.choice(HYPERPARAMETER_SPACE['conv_layers']),
        # Randomly select the learning rate
        'learning_rate': random.choice(HYPERPARAMETER_SPACE['learning_rates'])
    }

    # For each conv layer, generate layer-specific hyperparameters
    for i in range(individual['conv_layers']):
        # Number of filters in this layer
        individual[f'filters_{i}'] = random.choice(HYPERPARAMETER_SPACE['filters'])

        # Kernel size for this layer
        individual[f'kernel_size_{i}'] = random.choice(HYPERPARAMETER_SPACE['kernel_sizes'])

        # Activation function for this layer
        individual[f'activation_{i}'] = random.choice(HYPERPARAMETER_SPACE['activation_functions'])

        # Pooling type for this layer
        individual[f'pool_type_{i}'] = random.choice(HYPERPARAMETER_SPACE['pool_types'])

        # Dropout rate for this layer
        individual[f'dropout_{i}'] = random.choice(HYPERPARAMETER_SPACE['dropout_rates'])

    return individual


def initialize_population(population_size: int) -> List[Dict[str, Any]]:
    """
    Initialize a population of random individuals.

    Creates a list of randomly generated CNN architectures to form the
    initial population for the genetic algorithm.

    Args:
        population_size: The number of individuals to create

    Returns:
        List[Dict[str, Any]]: List of randomly generated individuals
    """
    return [generate_individual() for _ in range(population_size)]


def select_parents(
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        num_parents: int,
        tournament_size: int = 3
) -> List[Dict[str, Any]]:
    """
    Select parents using tournament selection.

    Tournament selection: randomly select k individuals and choose the best one.
    Repeat until we have num_parents parents.

    Args:
        population: List of individuals
        fitness_scores: Fitness scores corresponding to each individual
        num_parents: Number of parents to select
        tournament_size (int, optional): Number of individuals in each tournament.
                                         Large values increase selection pressure.
                                         Defaults to 3.

    Returns:
        List[Dict[str, Any]]: Selected parents
    """
    parents = []

    # Continue selecting parents until we have enough
    for _ in range(num_parents):
        # Randomly select tournament_size individuals for this tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)

        # Get the fitness scores for the selected individuals
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        # Find the winner (Select the individual from the tournament with the highest fitness)
        winner_relative_idx = np.argmax(tournament_fitness)
        winner_idx = tournament_indices[winner_relative_idx]

        # Add the winner to the parents list
        parents.append(population[winner_idx])

    return parents


def crossover(
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform crossover between two parents to create two offspring.

    This function combines hyperparameters from both parents to create two new
    CNN architecture configurations. The goal is to potentially create better
    architectures by combining good traits from both parents.

    Args:
        parent1 (Dict[str, Any]): First parent's hyperparameters
        parent2 (Dict[str, Any]): Second parent's hyperparameters

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Two new offspring
    """
    # Create copies to avoid modifying the originals
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Choose the smaller number of layers to simplify crossover
    min_layers = min(parent1['conv_layers'], parent2['conv_layers'])
    offspring1['conv_layers'] = min_layers
    offspring2['conv_layers'] = min_layers

    # Swap learning rates with 50% probability
    if random.random() < 0.5:
        offspring1['learning_rate'], offspring2['learning_rate'] = \
            offspring2['learning_rate'], offspring1['learning_rate']

    # For each layer, randomly decide whether to swap parameters
    for i in range(min_layers):
        for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
            param_name = f'{param}_{i}'
            if random.random() < 0.5:  # 50% chance to swap
                offspring1[param_name], offspring2[param_name] = \
                    offspring2[param_name], offspring1[param_name]

    # Occasionally allow offspring to have a different number of layers
    if random.random() < 0.2:  # 20% chance
        if random.random() < 0.5:
            # First offspring gets parent1's layer count
            if parent1['conv_layers'] > min_layers:
                offspring1['conv_layers'] = parent1['conv_layers']
                for i in range(min_layers, parent1['conv_layers']):
                    for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                        offspring1[f'{param}_{i}'] = parent1[f'{param}_{i}']
        else:
            # Second offspring gets parent2's layer count
            if parent2['conv_layers'] > min_layers:
                offspring2['conv_layers'] = parent2['conv_layers']
                for i in range(min_layers, parent2['conv_layers']):
                    for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                        offspring2[f'{param}_{i}'] = parent2[f'{param}_{i}']

    return offspring1, offspring2


def mutate(
        individual: Dict[str, Any],
        mutation_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Mutate an individual by randomly changing hyperparameters.

    The mutation_rate determines the probability of each hyperparameter being mutated.
    When a hyperparameter is selected for mutation, it is replaced with a random value
    from the corresponding search space.

    Args:
        individual (Dict[str, Any]): The individual to mutate
        mutation_rate (float, optional): Base probability of mutation.
                                         Default to 0.1.

    Returns:
        Dict[str, Any]: Mutated individual
    """
    # Create a copy to avoid modifying the original
    mutated = individual.copy()

    # Define mutation probability scaling factors
    scaling_factors = {
        'conv_layers': 0.5,  # Less likely to change architecture
        'learning_rate': 1.5,  # More likely to tune learning rate
        'filters': 1.2,  # Important but not drastic
        'kernel_size': 0.8,  # Less impact than filters
        'activation': 1.0,  # Normal importance
        'pool_type': 0.7,  # Less impact
        'dropout': 1.3  # Important for regularization
    }

    # Possibly mutate number of convolutional layers
    if random.random() < mutation_rate * scaling_factors['conv_layers']:
        new_conv_layers = random.choice(HYPERPARAMETER_SPACE['conv_layers'])

        # Handle decreasing layers
        if new_conv_layers < mutated['conv_layers']:
            for i in range(new_conv_layers, mutated['conv_layers']):
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    key = f'{param}_{i}'
                    if key in mutated:
                        del mutated[key]
        # Handle increasing layers
        elif new_conv_layers > mutated['conv_layers']:
            for i in range(mutated['conv_layers'], new_conv_layers):
                mutated[f'filters_{i}'] = random.choice(HYPERPARAMETER_SPACE['filters'])
                mutated[f'kernel_size_{i}'] = random.choice(HYPERPARAMETER_SPACE['kernel_sizes'])
                mutated[f'activation_{i}'] = random.choice(HYPERPARAMETER_SPACE['activation_functions'])
                mutated[f'pool_type_{i}'] = random.choice(HYPERPARAMETER_SPACE['pool_types'])
                mutated[f'dropout_{i}'] = random.choice(HYPERPARAMETER_SPACE['dropout_rates'])

        mutated['conv_layers'] = new_conv_layers

    # Possibly mutate learning rate
    if random.random() < mutation_rate * scaling_factors['learning_rate']:
        mutated['learning_rate'] = random.choice(HYPERPARAMETER_SPACE['learning_rates'])

    # Possibly mutate layer parameters
    for i in range(mutated['conv_layers']):
        # Mutate filters
        if random.random() < mutation_rate * scaling_factors['filters']:
            mutated[f'filters_{i}'] = random.choice(HYPERPARAMETER_SPACE['filters'])

        # Mutate kernel size
        if random.random() < mutation_rate * scaling_factors['kernel_size']:
            mutated[f'kernel_size_{i}'] = random.choice(HYPERPARAMETER_SPACE['kernel_sizes'])

        # Mutate activation function
        if random.random() < mutation_rate * scaling_factors['activation']:
            mutated[f'activation_{i}'] = random.choice(HYPERPARAMETER_SPACE['activation_functions'])

        # Mutate pool type
        if random.random() < mutation_rate * scaling_factors['pool_type']:
            mutated[f'pool_type_{i}'] = random.choice(HYPERPARAMETER_SPACE['pool_types'])

        # Mutate dropout rate
        if random.random() < mutation_rate * scaling_factors['dropout']:
            mutated[f'dropout_{i}'] = random.choice(HYPERPARAMETER_SPACE['dropout_rates'])

    return mutated


def evolve_population(
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        population_size: int,
        mutation_rate: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Evolve the population to the next generation.

    Args:
        population: Current generation of individuals
        fitness_scores: Fitness scores for current population
        population_size: Desired size of the new population
        mutation_rate: Probability of mutation

    Returns:
        List[Dict[str, Any]]: New generation of individuals
    """
    # Create a new empty generation
    next_generation = []

    # Apply elitism: Keep the best individual(s)
    elite_count = 1  # Number of top individuals to preserve
    if population_size >= 10:
        elite_count = 2  # Use more elites for larger populations

    # Find and add the elite individuals
    elite_indices = np.argsort(fitness_scores)[-elite_count:]
    for idx in elite_indices:
        next_generation.append(population[idx].copy())

    # Select parents for producing offspring
    num_parents = max(population_size - elite_count, 2)
    parents = select_parents(population, fitness_scores, num_parents)

    # Create offspring through crossover and mutation until we reach population_size
    while len(next_generation) < population_size:
        # Select two parents for crossover
        parent1, parent2 = random.sample(parents, 2)

        # Create offspring through crossover
        offspring1, offspring2 = crossover(parent1, parent2)

        # Apply mutation to offspring
        offspring1 = mutate(offspring1, mutation_rate)
        offspring2 = mutate(offspring2, mutation_rate)

        # Add first offspring to the next generation
        next_generation.append(offspring1)

        # Add second offspring if we still need more individuals
        if len(next_generation) < population_size:
            next_generation.append(offspring2)

    return next_generation


def run_genetic_algorithm(
        population_size: int = 10,
        num_generations: int = 10,
        mutation_rate: float = 0.1,
        epochs_per_eval: int = 5,
        batch_size: int = 64
) -> Tuple[Dict[str, Any], float, List[float]]:
    """
    Run the genetic algorithm to optimize CNN hyperparameters.

    Args:
        population_size: Number of individuals in each generation
        num_generations: Number of generations to evolve
        mutation_rate: Probability of mutation for each hyperparameter
        epochs_per_eval: Number of training epochs for each fitness evaluation
        batch_size: Batch size for training

    Returns:
        Tuple containing:
        - Best individual (hyperparameters)
        - Best fitness (validation accuracy)
        - Fitness history (list of best fitness values in each generation)
    """
    # Load data
    (x_train, y_train), (x_val, y_val), _ = load_data()

    # Initialize population
    population = initialize_population(population_size)

    # Track the best individual and fitness history
    best_individual = None
    best_fitness = 0
    fitness_history = []

    # Main loop
    for generation in range(num_generations):
        start_time = time.time()
        print(f"\nGeneration {generation + 1}/{num_generations}")

        # Evaluate fitness for each individual
        fitness_scores = []
        for i, individual in enumerate(population):
            print(f"  Evaluating individual {i + 1}/{population_size}...", end="\r")
            fitness = evaluate_fitness(
                individual, x_train, y_train, x_val, y_val,
                epochs=epochs_per_eval, batch_size=batch_size
            )
            fitness_scores.append(fitness)

        # Find the best individual in this generation
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_individual = population[gen_best_idx]

        # Update overall best if better
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual.copy()

        # Add to fitness history
        fitness_history.append(gen_best_fitness)

        # Print generation stats
        elapsed_time = time.time() - start_time
        print(f"  Best fitness in generation: {gen_best_fitness:.4f}")
        print(f"  Overall best fitness: {best_fitness:.4f}")
        print(f"  Time taken: {elapsed_time:.2f} seconds")

        # If we've reached the last generation, we're done
        if generation == num_generations - 1:
            break

        # Evolve population
        population = evolve_population(population, fitness_scores, population_size, mutation_rate)

    return best_individual, best_fitness, fitness_history
