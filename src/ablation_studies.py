# src/ablation_studies.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
import os
import json
import logging
from typing import Dict, Any, List, Tuple, Callable, Optional
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Import genetic algorithm components
from genetic_algorithms_starter import (
    load_data,
    generate_individual,
    initialize_population,
    evaluate_fitness,
    select_parents,
    crossover,
    mutate,
    evolve_population,
    run_genetic_algorithm
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ablation_studies.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AblationStudies")

# Define parameter variations for ablation studies
ABLATION_PARAMETERS = {
    "population_size": [5, 10, 20, 30],
    "num_generations": [5, 10, 15, 20],
    "mutation_rate": [0.05, 0.1, 0.2, 0.3],
    "tournament_size": [2, 3, 4, 5],
    "elitism_count": [1, 2, 3, 5]
}


# Define custom selection strategies
def tournament_selection(
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        num_parents: int,
        tournament_size: int = 3
) -> List[Dict[str, Any]]:
    """Tournament selection strategy."""
    parents = []

    for _ in range(num_parents):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_relative_idx = np.argmax(tournament_fitness)
        winner_idx = tournament_indices[winner_relative_idx]
        parents.append(population[winner_idx])

    return parents


def roulette_wheel_selection(
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        num_parents: int
) -> List[Dict[str, Any]]:
    """Roulette wheel (fitness proportionate) selection strategy."""
    parents = []

    # Calculate selection probabilities
    total_fitness = sum(fitness_scores)
    selection_probs = [f / total_fitness for f in fitness_scores] if total_fitness > 0 else None

    if selection_probs is None:
        # If all fitness scores are 0, select randomly
        return random.sample(population, num_parents)

    # Select parents based on probabilities
    for _ in range(num_parents):
        idx = np.random.choice(len(population), p=selection_probs)
        parents.append(population[idx])

    return parents


def rank_selection(
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        num_parents: int
) -> List[Dict[str, Any]]:
    """Rank-based selection strategy."""
    parents = []

    # Rank individuals from worst to best
    ranked_indices = np.argsort(fitness_scores)

    # Calculate selection probabilities based on rank
    n = len(population)
    ranks = np.arange(1, n + 1)  # Ranks from 1 to n
    total_rank = n * (n + 1) / 2  # Sum of ranks
    selection_probs = ranks / total_rank  # Higher rank = higher probability

    # Select parents based on rank probabilities
    for _ in range(num_parents):
        idx = np.random.choice(n, p=selection_probs)
        selected_idx = ranked_indices[idx]
        parents.append(population[selected_idx])

    return parents


# Define custom crossover strategies
def single_point_crossover(
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Single-point crossover strategy."""
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Choose minimum number of layers for simplicity
    min_layers = min(parent1['conv_layers'], parent2['conv_layers'])
    offspring1['conv_layers'] = min_layers
    offspring2['conv_layers'] = min_layers

    # Single crossover point
    if random.random() < 0.5:  # 50% chance to swap learning rates
        offspring1['learning_rate'], offspring2['learning_rate'] = \
            offspring2['learning_rate'], offspring1['learning_rate']

    # Pick a random crossover point for layer parameters
    if min_layers > 0:
        crossover_point = random.randint(0, min_layers - 1)

        # For each layer, swap parameters based on crossover point
        for i in range(min_layers):
            if i >= crossover_point:  # After crossover point, swap
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    param_name = f'{param}_{i}'
                    offspring1[param_name], offspring2[param_name] = \
                        offspring2[param_name], offspring1[param_name]

    return offspring1, offspring2


def uniform_crossover(
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Uniform crossover strategy."""
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Choose minimum number of layers for simplicity
    min_layers = min(parent1['conv_layers'], parent2['conv_layers'])
    offspring1['conv_layers'] = min_layers
    offspring2['conv_layers'] = min_layers

    # Swap learning rate with 50% probability
    if random.random() < 0.5:
        offspring1['learning_rate'], offspring2['learning_rate'] = \
            offspring2['learning_rate'], offspring1['learning_rate']

    # For each layer parameter, randomly decide which parent to inherit from
    for i in range(min_layers):
        for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
            param_name = f'{param}_{i}'
            if random.random() < 0.5:  # 50% chance to swap
                offspring1[param_name], offspring2[param_name] = \
                    offspring2[param_name], offspring1[param_name]

    return offspring1, offspring2


def whole_arithmetic_recombination(
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        alpha: float = 0.5
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Whole arithmetic recombination strategy.

    For numerical parameters, offspring are weighted averages of parents.
    For categorical parameters, they are selected randomly from parents.

    Args:
        parent1: First parent
        parent2: Second parent
        alpha: Weight for parent1 (1-alpha for parent2)
    """
    offspring1 = {}
    offspring2 = {}

    # For number of layers, take weighted average and round
    layers1 = parent1['conv_layers']
    layers2 = parent2['conv_layers']
    off1_layers = round(alpha * layers1 + (1 - alpha) * layers2)
    off2_layers = round((1 - alpha) * layers1 + alpha * layers2)

    offspring1['conv_layers'] = off1_layers
    offspring2['conv_layers'] = off2_layers

    # For learning rate, use weighted geometric mean
    lr1 = parent1['learning_rate']
    lr2 = parent2['learning_rate']
    offspring1['learning_rate'] = lr1 ** alpha * lr2 ** (1 - alpha)
    offspring2['learning_rate'] = lr1 ** (1 - alpha) * lr2 ** alpha

    # For layer parameters
    max_layers = max(off1_layers, off2_layers)

    for i in range(max_layers):
        # If this layer exists in both parents
        if i < min(parent1['conv_layers'], parent2['conv_layers']):
            # For numerical parameters (filters, kernel_size), use weighted average
            for param in ['filters', 'kernel_size']:
                param_name = f'{param}_{i}'
                val1 = parent1[param_name]
                val2 = parent2[param_name]

                # Round for integer values
                if i < off1_layers:
                    offspring1[param_name] = round(alpha * val1 + (1 - alpha) * val2)
                if i < off2_layers:
                    offspring2[param_name] = round((1 - alpha) * val1 + alpha * val2)

            # For dropout, use weighted average without rounding
            if i < off1_layers:
                offspring1[f'dropout_{i}'] = alpha * parent1[f'dropout_{i}'] + (1 - alpha) * parent2[f'dropout_{i}']
            if i < off2_layers:
                offspring2[f'dropout_{i}'] = (1 - alpha) * parent1[f'dropout_{i}'] + alpha * parent2[f'dropout_{i}']

            # For categorical parameters, randomly select from parents
            for param in ['activation', 'pool_type']:
                param_name = f'{param}_{i}'

                if i < off1_layers:
                    offspring1[param_name] = random.choice([parent1[param_name], parent2[param_name]])
                if i < off2_layers:
                    offspring2[param_name] = random.choice([parent1[param_name], parent2[param_name]])

        # If this layer only exists in parent1
        elif i < parent1['conv_layers']:
            if i < off1_layers:
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    offspring1[f'{param}_{i}'] = parent1[f'{param}_{i}']
            if i < off2_layers:
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    offspring2[f'{param}_{i}'] = parent1[f'{param}_{i}']

        # If this layer only exists in parent2
        elif i < parent2['conv_layers']:
            if i < off1_layers:
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    offspring1[f'{param}_{i}'] = parent2[f'{param}_{i}']
            if i < off2_layers:
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    offspring2[f'{param}_{i}'] = parent2[f'{param}_{i}']

    return offspring1, offspring2


# Define custom mutation strategies
def standard_mutation(
        individual: Dict[str, Any],
        mutation_rate: float = 0.1
) -> Dict[str, Any]:
    """Standard mutation strategy."""
    mutated = individual.copy()

    # Possibly mutate number of convolutional layers
    if random.random() < mutation_rate:
        # Get a new random number of layers from the search space
        new_conv_layers = random.choice([1, 2, 3, 4, 5])

        # Handle decreasing/increasing layers
        if new_conv_layers < mutated['conv_layers']:
            for i in range(new_conv_layers, mutated['conv_layers']):
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    key = f'{param}_{i}'
                    if key in mutated:
                        del mutated[key]
        elif new_conv_layers > mutated['conv_layers']:
            for i in range(mutated['conv_layers'], new_conv_layers):
                mutated[f'filters_{i}'] = random.choice([16, 32, 64, 128, 256])
                mutated[f'kernel_size_{i}'] = random.choice([3, 5, 7])
                mutated[f'activation_{i}'] = random.choice(['relu', 'elu', 'leaky_relu'])
                mutated[f'pool_type_{i}'] = random.choice(['max', 'avg', 'none'])
                mutated[f'dropout_{i}'] = random.choice([0.0, 0.25, 0.5])

        mutated['conv_layers'] = new_conv_layers

    # Possibly mutate learning rate
    if random.random() < mutation_rate:
        mutated['learning_rate'] = random.choice([0.1, 0.01, 0.001, 0.0001])

    # Possibly mutate layer parameters
    for i in range(mutated['conv_layers']):
        if random.random() < mutation_rate:
            mutated[f'filters_{i}'] = random.choice([16, 32, 64, 128, 256])
        if random.random() < mutation_rate:
            mutated[f'kernel_size_{i}'] = random.choice([3, 5, 7])
        if random.random() < mutation_rate:
            mutated[f'activation_{i}'] = random.choice(['relu', 'elu', 'leaky_relu'])
        if random.random() < mutation_rate:
            mutated[f'pool_type_{i}'] = random.choice(['max', 'avg', 'none'])
        if random.random() < mutation_rate:
            mutated[f'dropout_{i}'] = random.choice([0.0, 0.25, 0.5])

    return mutated


def gaussian_mutation(
        individual: Dict[str, Any],
        mutation_rate: float = 0.1,
        sigma: float = 0.2
) -> Dict[str, Any]:
    """
    Gaussian mutation strategy.

    For numerical parameters, add Gaussian noise.
    For categorical parameters, use standard mutation.
    """
    mutated = individual.copy()

    # Possibly mutate number of convolutional layers
    if random.random() < mutation_rate:
        # Add Gaussian noise and round
        new_layers = round(mutated['conv_layers'] + np.random.normal(0, sigma) * 2)
        # Ensure valid range
        new_layers = max(1, min(5, new_layers))

        # Handle decreasing/increasing layers
        if new_layers < mutated['conv_layers']:
            for i in range(new_layers, mutated['conv_layers']):
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    key = f'{param}_{i}'
                    if key in mutated:
                        del mutated[key]
        elif new_layers > mutated['conv_layers']:
            for i in range(mutated['conv_layers'], new_layers):
                mutated[f'filters_{i}'] = random.choice([16, 32, 64, 128, 256])
                mutated[f'kernel_size_{i}'] = random.choice([3, 5, 7])
                mutated[f'activation_{i}'] = random.choice(['relu', 'elu', 'leaky_relu'])
                mutated[f'pool_type_{i}'] = random.choice(['max', 'avg', 'none'])
                mutated[f'dropout_{i}'] = random.choice([0.0, 0.25, 0.5])

        mutated['conv_layers'] = new_layers

    # Possibly mutate learning rate
    if random.random() < mutation_rate:
        # Multiply by a random factor from a log-normal distribution
        factor = np.exp(np.random.normal(0, sigma))
        new_lr = mutated['learning_rate'] * factor
        # Ensure valid range
        new_lr = max(0.0001, min(0.1, new_lr))
        mutated['learning_rate'] = new_lr

    # Possibly mutate layer parameters
    for i in range(mutated['conv_layers']):
        # Mutate filters with Gaussian noise
        if random.random() < mutation_rate:
            current = mutated[f'filters_{i}']
            # Add noise scaled to the parameter's magnitude
            new_val = round(current * (1 + np.random.normal(0, sigma)))
            # Ensure valid range
            new_val = max(16, min(256, new_val))
            mutated[f'filters_{i}'] = new_val

        # Mutate kernel size with Gaussian noise
        if random.random() < mutation_rate:
            current = mutated[f'kernel_size_{i}']
            # Add small Gaussian noise
            new_val = round(current + np.random.normal(0, 1))
            # Ensure valid range and odd value
            new_val = max(3, min(7, new_val))
            # Make it odd
            if new_val % 2 == 0:
                new_val = new_val + 1 if new_val < 7 else new_val - 1
            mutated[f'kernel_size_{i}'] = new_val

        # For categorical parameters, use standard mutation
        if random.random() < mutation_rate:
            mutated[f'activation_{i}'] = random.choice(['relu', 'elu', 'leaky_relu'])
        if random.random() < mutation_rate:
            mutated[f'pool_type_{i}'] = random.choice(['max', 'avg', 'none'])

        # Mutate dropout with Gaussian noise
        if random.random() < mutation_rate:
            current = mutated[f'dropout_{i}']
            # Add small Gaussian noise
            new_val = current + np.random.normal(0, 0.1)
            # Ensure valid range
            new_val = max(0.0, min(0.5, new_val))
            mutated[f'dropout_{i}'] = new_val

    return mutated


def adaptive_mutation(
        individual: Dict[str, Any],
        mutation_rate: float = 0.1,
        generation: int = 0,
        max_generations: int = 10
) -> Dict[str, Any]:
    """
    Adaptive mutation strategy.

    Adjusts mutation based on generation progress:
    - Early generations: Higher mutation rate for exploration
    - Later generations: Lower mutation rate for exploitation
    """
    mutated = individual.copy()

    # Adjust mutation rate based on generation
    # Start with high rate, gradually decrease
    progress = generation / max_generations
    adaptive_rate = mutation_rate * (1.5 - progress)  # Linear decrease

    # Use standard mutation with adjusted rate
    return standard_mutation(individual, adaptive_rate)


# Define custom evolution strategies with elitism
def evolve_with_elitism(
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        population_size: int,
        mutation_rate: float = 0.1,
        elitism_count: int = 1,
        selection_func: Callable = tournament_selection,
        crossover_func: Callable = single_point_crossover,
        mutation_func: Callable = standard_mutation,
        **kwargs
) -> List[Dict[str, Any]]:
    """
    Evolve population with elitism.

    Args:
        population: Current population
        fitness_scores: Fitness scores
        population_size: Size of new population
        mutation_rate: Mutation rate
        elitism_count: Number of elite individuals to preserve
        selection_func: Function for parent selection
        crossover_func: Function for crossover
        mutation_func: Function for mutation
        **kwargs: Additional arguments for selection, crossover, mutation

    Returns:
        List[Dict[str, Any]]: New generation of individuals
    """
    # Create a new empty generation
    next_generation = []

    # Apply elitism: Keep the best individual(s)
    elite_indices = np.argsort(fitness_scores)[-elitism_count:]
    for idx in elite_indices:
        next_generation.append(population[idx].copy())

    # Select parents for producing offspring
    num_parents = max(population_size - elitism_count, 2)
    parents = selection_func(population, fitness_scores, num_parents, **kwargs)

    # Create offspring through crossover and mutation until we reach population_size
    while len(next_generation) < population_size:
        # Select two parents for crossover
        parent1, parent2 = random.sample(parents, 2)

        # Create offspring through crossover
        offspring1, offspring2 = crossover_func(parent1, parent2)

        # Apply mutation to offspring
        offspring1 = mutation_func(offspring1, mutation_rate, **kwargs)
        offspring2 = mutation_func(offspring2, mutation_rate, **kwargs)

        # Add first offspring to the next generation
        next_generation.append(offspring1)

        # Add second offspring if we still need more individuals
        if len(next_generation) < population_size:
            next_generation.append(offspring2)

    return next_generation


def run_ablation_study(
        selection_strategy: str = "tournament",
        crossover_strategy: str = "single_point",
        mutation_strategy: str = "standard",
        population_size: int = 10,
        num_generations: int = 10,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        elitism_count: int = 1,
        epochs_per_eval: int = 3,
        run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run genetic algorithm with specific parameters for ablation study.

    Args:
        selection_strategy: Parent selection strategy
        crossover_strategy: Crossover strategy
        mutation_strategy: Mutation strategy
        population_size: Population size
        num_generations: Number of generations
        mutation_rate: Mutation rate
        tournament_size: Tournament size for tournament selection
        elitism_count: Number of elite individuals preserved
        epochs_per_eval: Epochs per fitness evaluation
        run_id: Optional identifier for this run

    Returns:
        Dict[str, Any]: Results including the best individual, fitness, and history
    """
    # Set up logging prefix
    prefix = f"[Run {run_id}] " if run_id else ""
    logger.info(f"{prefix}Starting ablation study run with:")
    logger.info(f"{prefix}  Selection: {selection_strategy}")
    logger.info(f"{prefix}  Crossover: {crossover_strategy}")
    logger.info(f"{prefix}  Mutation: {mutation_strategy}")
    logger.info(f"{prefix}  Population: {population_size}")
    logger.info(f"{prefix}  Generations: {num_generations}")
    logger.info(f"{prefix}  Mutation rate: {mutation_rate}")
    logger.info(f"{prefix}  Tournament size: {tournament_size}")
    logger.info(f"{prefix}  Elitism count: {elitism_count}")

    # Create results directory if it doesn't exist
    os.makedirs("results/ablation", exist_ok=True)

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Select appropriate functions based on strategies
    selection_funcs = {
        "tournament": tournament_selection,
        "roulette": roulette_wheel_selection,
        "rank": rank_selection
    }

    crossover_funcs = {
        "single_point": single_point_crossover,
        "uniform": uniform_crossover,
        "arithmetic": whole_arithmetic_recombination
    }

    mutation_funcs = {
        "standard": standard_mutation,
        "gaussian": gaussian_mutation,
        "adaptive": adaptive_mutation
    }

    selection_func = selection_funcs.get(selection_strategy, tournament_selection)
    crossover_func = crossover_funcs.get(crossover_strategy, single_point_crossover)
    mutation_func = mutation_funcs.get(mutation_strategy, standard_mutation)

    # Initialize population
    population = initialize_population(population_size)

    # Track the best individual and fitness history
    best_individual = None
    best_fitness = 0
    fitness_history = []
    population_fitness_histories = []

    # Prepare mutation arguments
    mutation_args = {}
    if mutation_strategy == "adaptive":
        mutation_args["max_generations"] = num_generations

    # Main evolution loop
    start_time = time.time()
    for generation in range(num_generations):
        generation_start = time.time()
        logger.info(f"{prefix}Generation {generation + 1}/{num_generations}")

        # Update mutation arguments for adaptive mutation
        if mutation_strategy == "adaptive":
            mutation_args["generation"] = generation

        # Evaluate fitness for each individual
        fitness_scores = []
        for i, individual in enumerate(population):
            fitness = evaluate_fitness(
                individual, x_train, y_train, x_val, y_val,
                epochs=epochs_per_eval
            )
            fitness_scores.append(fitness)

        # Save population fitness for this generation
        population_fitness_histories.append(fitness_scores.copy())

        # Find the best individual in this generation
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_individual = population[gen_best_idx]

        # Update overall best if better
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual.copy()
            logger.info(f"{prefix}  New best individual! Fitness: {best_fitness:.4f}")

        # Add to fitness history
        fitness_history.append(gen_best_fitness)

        # Log generation stats
        generation_time = time.time() - generation_start
        logger.info(f"{prefix}  Best fitness: {gen_best_fitness:.4f}")
        logger.info(f"{prefix}  Avg fitness: {np.mean(fitness_scores):.4f}")
        logger.info(f"{prefix}  Generation time: {generation_time:.2f} seconds")

        # If we've reached the last generation, we're done
        if generation == num_generations - 1:
            break

        # Evolve population for next generation
        # If using tournament selection, pass tournament_size
        selection_kwargs = {}
        if selection_strategy == "tournament":
            selection_kwargs["tournament_size"] = tournament_size

        # If using arithmetic crossover, pass alpha parameter
        crossover_kwargs = {}
        if crossover_strategy == "arithmetic":
            crossover_kwargs["alpha"] = 0.5  # Default alpha

        # Evolve population
        population = evolve_with_elitism(
            population=population,
            fitness_scores=fitness_scores,
            population_size=population_size,
            mutation_rate=mutation_rate,
            elitism_count=elitism_count,
            selection_func=selection_func,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            **selection_kwargs,
            **crossover_kwargs,
            **mutation_args
        )

    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"{prefix}Ablation study completed in {total_time:.2f} seconds")
    logger.info(f"{prefix}Best fitness achieved: {best_fitness:.4f}")

    # Prepare and return results
    results = {
        "selection_strategy": selection_strategy,
        "crossover_strategy": crossover_strategy,
        "mutation_strategy": mutation_strategy,
        "population_size": population_size,
        "num_generations": num_generations,
        "mutation_rate": mutation_rate,
        "tournament_size": tournament_size,
        "elitism_count": elitism_count,
        "epochs_per_eval": epochs_per_eval,
        "best_fitness": best_fitness,
        "best_individual": best_individual,
        "fitness_history": fitness_history,
        "population_fitness_histories": population_fitness_histories,
        "total_time": total_time,
        "run_id": run_id
    }

    # Save results to file
    result_filename = f"ablation_{selection_strategy}_{crossover_strategy}_{mutation_strategy}"
    result_filename += f"_pop{population_size}_gen{num_generations}_mut{mutation_rate}"
    result_filename += f"_ts{tournament_size}_el{elitism_count}"
    if run_id:
        result_filename += f"_{run_id}"

    with open(f"results/ablation/{result_filename}.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = results.copy()
        serializable_results["fitness_history"] = [float(x) for x in fitness_history]
        serializable_results["population_fitness_histories"] = [
            [float(x) for x in gen_scores] for gen_scores in population_fitness_histories
        ]
        json.dump(serializable_results, f, indent=2)

    return results


def worker_func(params):
    """Worker function for parallel ablation studies."""
    return run_ablation_study(**params)


def run_multiple_ablation_studies(parallel: bool = True):
    """
    Run multiple ablation studies to test different parameters.

    Args:
        parallel: Whether to run studies in parallel
    """
    logger.info("Starting multiple ablation studies...")

    # Define parameter combinations to test
    studies = [
        # Test different selection strategies
        {"selection_strategy": "tournament", "run_id": "select_tournament"},
        {"selection_strategy": "roulette", "run_id": "select_roulette"},
        {"selection_strategy": "rank", "run_id": "select_rank"},

        # Test different crossover strategies
        {"crossover_strategy": "single_point", "run_id": "cross_single"},
        {"crossover_strategy": "uniform", "run_id": "cross_uniform"},
        {"crossover_strategy": "arithmetic", "run_id": "cross_arithmetic"},

        # Test different mutation strategies
        {"mutation_strategy": "standard", "run_id": "mut_standard"},
        {"mutation_strategy": "gaussian", "run_id": "mut_gaussian"},
        {"mutation_strategy": "adaptive", "run_id": "mut_adaptive"},

        # Test different population sizes
        {"population_size": 5, "run_id": "pop_5"},
        {"population_size": 10, "run_id": "pop_10"},
        {"population_size": 20, "run_id": "pop_20"},

        # Test different generation counts
        {"num_generations": 5, "run_id": "gen_5"},
        {"num_generations": 10, "run_id": "gen_10"},
        {"num_generations": 15, "run_id": "gen_15"},

        # Test different mutation rates
        {"mutation_rate": 0.05, "run_id": "mut_rate_0.05"},
        {"mutation_rate": 0.1, "run_id": "mut_rate_0.1"},
        {"mutation_rate": 0.2, "run_id": "mut_rate_0.2"},

        # Test different tournament sizes
        {"tournament_size": 2, "run_id": "tourney_2"},
        {"tournament_size": 3, "run_id": "tourney_3"},
        {"tournament_size": 5, "run_id": "tourney_5"},

        # Test different elitism counts
        {"elitism_count": 1, "run_id": "elite_1"},
        {"elitism_count": 2, "run_id": "elite_2"},
        {"elitism_count": 3, "run_id": "elite_3"}
    ]

    # Set default parameters
    default_params = {
        "selection_strategy": "tournament",
        "crossover_strategy": "single_point",
        "mutation_strategy": "standard",
        "population_size": 10,
        "num_generations": 5,  # Use fewer generations for ablation to save time
        "mutation_rate": 0.1,
        "tournament_size": 3,
        "elitism_count": 1,
        "epochs_per_eval": 2  # Use fewer epochs for ablation to save time
    }

    # Complete each study with default parameters
    for study in studies:
        for param, value in default_params.items():
            if param not in study:
                study[param] = value

    # Run studies in parallel or sequentially
    results = []

    if parallel and __name__ == "__main__":
        # Determine the number of processes
        num_processes = min(multiprocessing.cpu_count(), len(studies))
        logger.info(f"Running {len(studies)} ablation studies in parallel with {num_processes} processes")

        # Run in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(worker_func, studies))
    else:
        logger.info(f"Running {len(studies)} ablation studies sequentially")

        # Run sequentially
        for study in tqdm(studies, desc="Ablation Studies"):
            results.append(run_ablation_study(**study))

    logger.info("All ablation studies completed!")

    # Analyze and visualize results
    visualize_ablation_results()


def visualize_ablation_results():
    """Visualize and analyze the results of ablation studies."""
    logger.info("Visualizing ablation study results...")

    # Create results directory if it doesn't exist
    os.makedirs("results/ablation", exist_ok=True)

    # Load all ablation study results
    results = []
    result_files = [f for f in os.listdir("results/ablation") if f.endswith(".json")]

    if not result_files:
        logger.warning("No ablation study results found to visualize")
        return

    for file in result_files:
        try:
            with open(f"results/ablation/{file}", "r") as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    logger.info(f"Loaded {len(results)} ablation study results")

    # Group results by parameter type
    param_groups = {
        "selection_strategy": [],
        "crossover_strategy": [],
        "mutation_strategy": [],
        "population_size": [],
        "num_generations": [],
        "mutation_rate": [],
        "tournament_size": [],
        "elitism_count": []
    }

    # Organize results by parameter groups
    for result in results:
        for param in param_groups.keys():
            run_id = result.get("run_id", "")

            # Check if this result is testing this parameter
            if param in run_id or param.replace("_", "") in run_id:
                param_groups[param].append(result)

    # Visualize each parameter group
    for param, group_results in param_groups.items():
        if not group_results:
            continue

        logger.info(f"Visualizing impact of {param}...")

        # Extract parameter values and best fitness
        param_values = []
        best_fitness = []
        convergence_speed = []  # Measure of how quickly fitness improves

        for result in group_results:
            # Extract parameter value
            param_value = result[param]
            param_values.append(str(param_value))

            # Extract the best fitness
            best_fitness.append(result["best_fitness"])

            # Calculate convergence speed (fitness improvement in first 3 generations)
            if len(result["fitness_history"]) >= 3:
                initial_fitness = result["fitness_history"][0]
                third_gen_fitness = result["fitness_history"][2]
                speed = third_gen_fitness - initial_fitness
                convergence_speed.append(speed)
            else:
                convergence_speed.append(0)

        # Sort by parameter value if numerical
        if param in ["population_size", "num_generations", "mutation_rate",
                     "tournament_size", "elitism_count"]:
            sorted_indices = np.argsort([float(v) for v in param_values])
            param_values = [param_values[i] for i in sorted_indices]
            best_fitness = [best_fitness[i] for i in sorted_indices]
            if convergence_speed:
                convergence_speed = [convergence_speed[i] for i in sorted_indices]

        # Create bar chart for the best fitness
        plt.figure(figsize=(10, 6))
        bars = plt.bar(param_values, best_fitness, color='skyblue')

        # Add value labels
        for bar, value in zip(bars, best_fitness):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.4f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

        plt.xlabel(param.replace("_", " ").title())
        plt.ylabel("Best Fitness (Validation Accuracy)")
        plt.title(f"Impact of {param.replace('_', ' ').title()} on Best Fitness")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save figure
        plt.savefig(f"results/ablation/ablation_{param}_fitness.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Create a line plot for fitness history
        plt.figure(figsize=(12, 6))

        # Plot fitness history for each result in this group
        for i, result in enumerate(group_results):
            label = f"{param}={result[param]}"
            plt.plot(range(1, len(result["fitness_history"]) + 1),
                     result["fitness_history"],
                     marker='o',
                     label=label)

        plt.xlabel("Generation")
        plt.ylabel("Best Fitness (Validation Accuracy)")
        plt.title(f"Fitness Convergence for Different {param.replace('_', ' ').title()} Values")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save figure
        plt.savefig(f"results/ablation/ablation_{param}_convergence.png", dpi=150, bbox_inches='tight')
        plt.close()

        # If we have convergence speed, create a bar chart
        if convergence_speed:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(param_values, convergence_speed, color='lightgreen')

            # Add value labels
            for bar, value in zip(bars, convergence_speed):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01 if value >= 0 else value - 0.02,
                    f"{value:.4f}",
                    ha='center',
                    va='bottom' if value >= 0 else 'top',
                    fontsize=9
                )

            plt.xlabel(param.replace("_", " ").title())
            plt.ylabel("Convergence Speed (Fitness Improvement in First 3 Generations)")
            plt.title(f"Impact of {param.replace('_', ' ').title()} on Convergence Speed")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save figure
            plt.savefig(f"results/ablation/ablation_{param}_speed.png", dpi=150, bbox_inches='tight')
            plt.close()

    # Create a summary table comparing all strategies
    create_ablation_summary_table(results)

    logger.info("Ablation study visualization complete")


def create_ablation_summary_table(results: List[Dict[str, Any]]):
    """
    Create a summary table comparing all ablation studies.

    Args:
        results: List of ablation study results
    """
    # Create DataFrame for summary
    summary_data = []

    for result in results:
        summary_data.append({
            "Selection": result["selection_strategy"],
            "Crossover": result["crossover_strategy"],
            "Mutation": result["mutation_strategy"],
            "Population": result["population_size"],
            "Generations": result["num_generations"],
            "Mutation Rate": result["mutation_rate"],
            "Tournament Size": result["tournament_size"],
            "Elitism": result["elitism_count"],
            "Best Fitness": result["best_fitness"],
            "Runtime (s)": result["total_time"],
            "Run ID": result.get("run_id", "")
        })

    df = pd.DataFrame(summary_data)

    # Sort by the best fitness
    df = df.sort_values("Best Fitness", ascending=False)

    # Save to CSV
    df.to_csv("results/ablation/ablation_summary.csv", index=False)

    # Create a visual table for the top results
    top_results = df.head(10)

    plt.figure(figsize=(14, 8))
    plt.axis('off')

    # Create table
    table = plt.table(
        cellText=top_results.values,
        colLabels=top_results.columns,
        cellLoc='center',
        loc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color the header row
    for j, cell in enumerate(table._cells[(0, j)] for j in range(len(top_results.columns))):
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')

    plt.title("Top 10 Ablation Study Results", fontsize=16, pad=20)
    plt.tight_layout()

    # Save figure
    plt.savefig("results/ablation/ablation_top_results.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_focused_ablation_study():
    """
    Run a more focused ablation study based on what we've learned.
    Tests combinations of the best parameters found so far.
    """
    logger.info("Running focused ablation study...")

    # Define combinations of the best parameters to test
    studies = [
        {
            "selection_strategy": "tournament",
            "crossover_strategy": "uniform",
            "mutation_strategy": "adaptive",
            "population_size": 20,
            "num_generations": 10,
            "mutation_rate": 0.1,
            "tournament_size": 3,
            "elitism_count": 2,
            "epochs_per_eval": 3,
            "run_id": "best_combo_1"
        },
        {
            "selection_strategy": "rank",
            "crossover_strategy": "arithmetic",
            "mutation_strategy": "gaussian",
            "population_size": 15,
            "num_generations": 8,
            "mutation_rate": 0.15,
            "tournament_size": 4,
            "elitism_count": 1,
            "epochs_per_eval": 3,
            "run_id": "best_combo_2"
        },
        # Add more combinations as needed
    ]

    # Run studies sequentially
    results = []
    for study in studies:
        results.append(run_ablation_study(**study))

    # Create comparison visualization
    visualize_ablation_results()


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation studies for genetic algorithm")
    parser.add_argument("--single", action="store_true", help="Run a single ablation study")
    parser.add_argument("--multiple", action="store_true", help="Run multiple ablation studies")
    parser.add_argument("--focused", action="store_true", help="Run focused ablation study")
    parser.add_argument("--visualize", action="store_true", help="Visualize existing ablation results")
    parser.add_argument("--sequential", action="store_true", help="Run studies sequentially (not in parallel)")

    args = parser.parse_args()

    if args.single:
        # Run a single ablation study with default parameters
        run_ablation_study()
    elif args.multiple:
        # Run multiple ablation studies
        run_multiple_ablation_studies(not args.sequential)
    elif args.focused:
        # Run focused ablation study
        run_focused_ablation_study()
    elif args.visualize:
        # Visualize existing results
        visualize_ablation_results()
    else:
        # Default: run a single study
        logger.info("No option specified, running a single ablation study")
        run_ablation_study()
