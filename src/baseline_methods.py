import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
import itertools
from typing import Dict, Any, List, Tuple, Callable
import os
import logging
from tqdm import tqdm
import json

# Import our genetic algorithm components
from genetic_algorithms_starter import (
    load_data,
    build_model,
    evaluate_fitness,
    HYPERPARAMETER_SPACE,
    run_genetic_algorithm
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("baseline_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BaselineMethods")


def run_random_search(
        num_samples: int = 20,
        epochs_per_eval: int = 3
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]], List[float]]:
    """
    Perform random search for hyperparameter tuning.

    Args:
        num_samples: Number of random hyperparameter combinations to try
        epochs_per_eval: Number of training epochs per fitness evaluation

    Returns:
        Tuple containing:
        - Best individual (hyperparameters)
        - Best fitness (validation accuracy)
        - All individuals evaluated
        - All fitness scores
    """
    logger.info(f"Starting random search with {num_samples} samples")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load data
    logger.info("Loading data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    logger.info(f"Data loaded: {x_train.shape[0]} training, {x_val.shape[0]} validation samples")

    # Track all individuals and their fitness scores
    all_individuals = []
    all_scores = []

    # Track the best individual
    best_individual = None
    best_fitness = 0

    # Random search loop
    start_time = time.time()
    for i in tqdm(range(num_samples), desc="Random Search Progress"):
        # Generate a random hyperparameter set
        individual = generate_random_individual()

        # Evaluate the individual
        fitness = evaluate_fitness(
            individual, x_train, y_train, x_val, y_val,
            epochs=epochs_per_eval
        )

        # Track this individual
        all_individuals.append(individual)
        all_scores.append(fitness)

        # Update best if better
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual.copy()
            logger.info(f"New best individual found (sample {i + 1}/{num_samples}): Fitness = {best_fitness:.4f}")

    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Random search completed in {total_time:.2f} seconds")
    logger.info(f"Best fitness achieved: {best_fitness:.4f}")

    # Save results
    results = {
        "method": "random_search",
        "num_samples": num_samples,
        "epochs_per_eval": epochs_per_eval,
        "best_fitness": best_fitness,
        "best_individual": best_individual,
        "total_time": total_time,
        "fitness_scores": all_scores
    }

    with open("results/random_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_individual, best_fitness, all_individuals, all_scores


def create_grid_search_space(
        max_combinations: int = 100
) -> List[Dict[str, Any]]:
    """
    Create a grid of hyperparameter combinations for grid search.
    Limit the search space to avoid combinatorial explosion.

    Args:
        max_combinations: Maximum number of combinations to generate

    Returns:
        List of hyperparameter combinations (individuals)
    """
    # Define a reduced search space to keep the grid manageable
    reduced_space = {
        'conv_layers': [1, 2, 3],  # Reduced from [1, 2, 3, 4, 5]
        'filters': [32, 64, 128],  # Reduced from [16, 32, 64, 128, 256]
        'kernel_sizes': [3, 5],  # Reduced from [3, 5, 7]
        'pool_types': ['max', 'avg'],  # Reduced from ['max', 'avg', 'none']
        'learning_rates': [0.01, 0.001],  # Reduced from [0.1, 0.01, 0.001, 0.0001]
        'activation_functions': ['relu', 'elu'],  # Reduced from ['relu', 'elu', 'leaky_relu']
        'dropout_rates': [0.0, 0.25]  # Reduced from [0.0, 0.25, 0.5]
    }

    # Calculate total combinations without layer constraints
    # This would be a full factorial design
    total_layer_params = len(reduced_space['filters']) * len(reduced_space['kernel_sizes']) * len(reduced_space['activation_functions']) * len(reduced_space['pool_types']) * len(reduced_space['dropout_rates'])

    max_layers = min(reduced_space['conv_layers'][-1],
                     int(np.log(max_combinations) / np.log(total_layer_params)))

    # Further reduce the number of layers if needed
    if max_layers < reduced_space['conv_layers'][-1]:
        reduced_space['conv_layers'] = list(range(1, max_layers + 1))

    logger.info(f"Grid search space reduced to max {max_layers} layers")

    # Generate grid combinations
    grid_combinations = []

    # For each number of layers
    for num_layers in reduced_space['conv_layers']:
        # For each learning rate
        for lr in reduced_space['learning_rates']:
            # Generate all combinations for layer parameters
            # To avoid combinatorial explosion, we'll use the same parameters for all layers
            for filters in reduced_space['filters']:
                for kernel_size in reduced_space['kernel_sizes']:
                    for activation in reduced_space['activation_functions']:
                        for pool_type in reduced_space['pool_types']:
                            for dropout in reduced_space['dropout_rates']:
                                # Create individual
                                individual = {
                                    'conv_layers': num_layers,
                                    'learning_rate': lr
                                }

                                # Add layer-specific parameters
                                for i in range(num_layers):
                                    individual[f'filters_{i}'] = filters
                                    individual[f'kernel_size_{i}'] = kernel_size
                                    individual[f'activation_{i}'] = activation
                                    individual[f'pool_type_{i}'] = pool_type
                                    individual[f'dropout_{i}'] = dropout

                                grid_combinations.append(individual)

    # Check if we need to further reduce combinations
    if len(grid_combinations) > max_combinations:
        logger.warning(f"Grid has {len(grid_combinations)} combinations, reducing to {max_combinations}")
        # Randomly sample max_combinations from the grid
        grid_combinations = random.sample(grid_combinations, max_combinations)

    logger.info(f"Created grid with {len(grid_combinations)} hyperparameter combinations")
    return grid_combinations


def run_grid_search(
        max_combinations: int = 100,
        epochs_per_eval: int = 3
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]], List[float]]:
    """
    Perform grid search for hyperparameter tuning.

    Args:
        max_combinations: Maximum number of hyperparameter combinations to try
        epochs_per_eval: Number of training epochs per fitness evaluation

    Returns:
        Tuple containing:
        - Best individual (hyperparameters)
        - Best fitness (validation accuracy)
        - All individuals evaluated
        - All fitness scores
    """
    logger.info(f"Starting grid search with max {max_combinations} combinations")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load data
    logger.info("Loading data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    logger.info(f"Data loaded: {x_train.shape[0]} training, {x_val.shape[0]} validation samples")

    # Create grid search space
    grid_individuals = create_grid_search_space(max_combinations)

    # Track all individuals and their fitness scores
    all_individuals = []
    all_scores = []

    # Track the best individual
    best_individual = None
    best_fitness = 0

    # Grid search loop
    start_time = time.time()
    for i, individual in enumerate(tqdm(grid_individuals, desc="Grid Search Progress")):
        # Evaluate the individual
        fitness = evaluate_fitness(
            individual, x_train, y_train, x_val, y_val,
            epochs=epochs_per_eval
        )

        # Track this individual
        all_individuals.append(individual)
        all_scores.append(fitness)

        # Update best if better
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual.copy()
            logger.info(
                f"New best individual found (combination {i + 1}/{len(grid_individuals)}): Fitness = {best_fitness:.4f}")

    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Grid search completed in {total_time:.2f} seconds")
    logger.info(f"Best fitness achieved: {best_fitness:.4f}")

    # Save results
    results = {
        "method": "grid_search",
        "max_combinations": max_combinations,
        "actual_combinations": len(grid_individuals),
        "epochs_per_eval": epochs_per_eval,
        "best_fitness": best_fitness,
        "best_individual": best_individual,
        "total_time": total_time,
        "fitness_scores": all_scores
    }

    with open("results/grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_individual, best_fitness, all_individuals, all_scores


def generate_random_individual() -> Dict[str, Any]:
    """
    Generate a random individual (hyperparameter set).
    This is similar to the function in genetic_algorithms_starter.py.

    Returns:
        Dict[str, Any]: Randomly generated hyperparameters
    """
    # Start with basic hyperparameters
    individual = {
        'conv_layers': random.choice(HYPERPARAMETER_SPACE['conv_layers']),
        'learning_rate': random.choice(HYPERPARAMETER_SPACE['learning_rates'])
    }

    # Add layer-specific hyperparameters
    for i in range(individual['conv_layers']):
        individual[f'filters_{i}'] = random.choice(HYPERPARAMETER_SPACE['filters'])
        individual[f'kernel_size_{i}'] = random.choice(HYPERPARAMETER_SPACE['kernel_sizes'])
        individual[f'activation_{i}'] = random.choice(HYPERPARAMETER_SPACE['activation_functions'])
        individual[f'pool_type_{i}'] = random.choice(HYPERPARAMETER_SPACE['pool_types'])
        individual[f'dropout_{i}'] = random.choice(HYPERPARAMETER_SPACE['dropout_rates'])

    return individual


def run_genetic_algorithm_for_comparison(
        population_size: int = 10,
        num_generations: int = 10,
        mutation_rate: float = 0.1,
        epochs_per_eval: int = 3
) -> Dict:
    """
    Run the genetic algorithm and format results for comparison.

    Args:
        population_size: Size of the population
        num_generations: Number of generations to evolve
        mutation_rate: Probability of mutation
        epochs_per_eval: Number of training epochs per fitness evaluation

    Returns:
        Dict: Results including the best individual, fitness, and timing
    """
    logger.info("Running genetic algorithm for comparison...")

    start_time = time.time()
    best_individual, best_fitness, fitness_history = run_genetic_algorithm(
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        epochs_per_eval=epochs_per_eval
    )
    total_time = time.time() - start_time

    # Calculate total evaluations
    total_evaluations = population_size * num_generations

    logger.info(f"Genetic algorithm completed in {total_time:.2f} seconds")
    logger.info(f"Best fitness achieved: {best_fitness:.4f}")

    # Save results
    results = {
        "method": "genetic_algorithm",
        "population_size": population_size,
        "num_generations": num_generations,
        "mutation_rate": mutation_rate,
        "epochs_per_eval": epochs_per_eval,
        "best_fitness": best_fitness,
        "best_individual": best_individual,
        "total_time": total_time,
        "fitness_history": fitness_history,
        "total_evaluations": total_evaluations
    }

    with open("results/genetic_algorithm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_comparison_results(
        ga_results: Dict,
        random_results: Dict,
        grid_results: Dict,
        filename: str = "hyperparameter_tuning_comparison"
):
    """
    Create comparison plots of the different hyperparameter tuning methods.

    Args:
        ga_results: Results from genetic algorithm
        random_results: Results from random search
        grid_results: Results from grid search
        filename: Base filename for the output images
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Plot 1: Compare the best fitness
    plt.figure(figsize=(12, 6))
    methods = ['Genetic Algorithm', 'Random Search', 'Grid Search']
    best_fitness = [ga_results['best_fitness'], random_results['best_fitness'], grid_results['best_fitness']]

    # Create bar chart
    plt.bar(methods, best_fitness, color=['#3498db', '#e74c3c', '#2ecc71'])

    # Add values above bars
    for i, v in enumerate(best_fitness):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.ylabel('Best Validation Accuracy')
    plt.title('Comparison of Best Validation Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save plot
    plt.savefig(f"results/{filename}_best_fitness.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Compare runtime efficiency
    plt.figure(figsize=(12, 6))

    # Calculate time per evaluation for each method
    ga_time_per_eval = ga_results['total_time'] / ga_results['total_evaluations']
    random_time_per_eval = random_results['total_time'] / len(random_results['fitness_scores'])
    grid_time_per_eval = grid_results['total_time'] / len(grid_results['fitness_scores'])

    time_per_eval = [ga_time_per_eval, random_time_per_eval, grid_time_per_eval]

    # Create bar chart
    plt.bar(methods, time_per_eval, color=['#3498db', '#e74c3c', '#2ecc71'])

    # Add values above bars
    for i, v in enumerate(time_per_eval):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')

    plt.ylabel('Time per Evaluation (seconds)')
    plt.title('Comparison of Computational Efficiency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save plot
    plt.savefig(f"results/{filename}_time_efficiency.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Compare total runtime and evaluations
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for total runtime
    total_times = [ga_results['total_time'], random_results['total_time'], grid_results['total_time']]
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, total_times, width, label='Total Runtime (s)', color='#3498db')
    ax1.set_ylabel('Total Runtime (seconds)', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')

    # Add values above bars
    for i, v in enumerate(total_times):
        ax1.text(i - width / 2, v + max(total_times) * 0.05, f"{v:.1f}s", ha='center', color='#3498db')

    # Second y-axis for total evaluations
    ax2 = ax1.twinx()
    total_evals = [
        ga_results['total_evaluations'],
        len(random_results['fitness_scores']),
        len(grid_results['fitness_scores'])
    ]

    bars2 = ax2.bar(x + width / 2, total_evals, width, label='Total Evaluations', color='#e74c3c')
    ax2.set_ylabel('Total Evaluations', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    # Add values above bars
    for i, v in enumerate(total_evals):
        ax2.text(i + width / 2, v + max(total_evals) * 0.05, f"{v}", ha='center', color='#e74c3c')

    # Add legend and title
    fig.tight_layout()
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    fig.legend(handles=[bars1, bars2], loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=2)
    plt.title('Comparison of Computational Cost', pad=30)

    # Save plot
    plt.savefig(f"results/{filename}_computational_cost.png", dpi=150, bbox_inches='tight')
    plt.close()

    # If genetic algorithm has fitness history, plot convergence
    if 'fitness_history' in ga_results:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(ga_results['fitness_history']) + 1),
                 ga_results['fitness_history'],
                 'b-', marker='o')
        plt.axhline(y=random_results['best_fitness'], color='r', linestyle='--',
                    label=f'Random Search Best: {random_results["best_fitness"]:.4f}')
        plt.axhline(y=grid_results['best_fitness'], color='g', linestyle='--',
                    label=f'Grid Search Best: {grid_results["best_fitness"]:.4f}')

        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (Validation Accuracy)')
        plt.title('Genetic Algorithm Convergence vs. Baseline Methods')
        plt.grid(True)
        plt.legend()

        # Save plot
        plt.savefig(f"results/{filename}_ga_convergence.png", dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Comparison plots saved to results/{filename}_*.png")


def run_comparison(
        ga_population_size: int = 10,
        ga_num_generations: int = 5,
        random_samples: int = 50,
        grid_max_combinations: int = 50,
        epochs_per_eval: int = 3
):
    """
    Run and compare all hyperparameter tuning methods.

    Args:
        ga_population_size: Population size for genetic algorithm
        ga_num_generations: Number of generations for genetic algorithm
        random_samples: Number of samples for random search
        grid_max_combinations: Maximum combinations for grid search
        epochs_per_eval: Number of training epochs per fitness evaluation
    """
    logger.info("Starting hyperparameter tuning methods comparison")

    # Calculate total evaluations for genetic algorithm
    ga_total_evals = ga_population_size * ga_num_generations

    # Log comparison setup
    logger.info(
        f"Genetic Algorithm: {ga_population_size} population × {ga_num_generations} generations = {ga_total_evals} evaluations")
    logger.info(f"Random Search: {random_samples} evaluations")
    logger.info(f"Grid Search: max {grid_max_combinations} evaluations")
    logger.info(f"Each evaluation trains for {epochs_per_eval} epochs")

    # Run random search
    logger.info("\n" + "=" * 50)
    logger.info("Running Random Search")
    logger.info("=" * 50)
    random_best, random_fitness, random_all, random_scores = run_random_search(
        num_samples=random_samples,
        epochs_per_eval=epochs_per_eval
    )

    # Run grid search
    logger.info("\n" + "=" * 50)
    logger.info("Running Grid Search")
    logger.info("=" * 50)
    grid_best, grid_fitness, grid_all, grid_scores = run_grid_search(
        max_combinations=grid_max_combinations,
        epochs_per_eval=epochs_per_eval
    )

    # Run genetic algorithm
    logger.info("\n" + "=" * 50)
    logger.info("Running Genetic Algorithm")
    logger.info("=" * 50)
    ga_results = run_genetic_algorithm_for_comparison(
        population_size=ga_population_size,
        num_generations=ga_num_generations,
        mutation_rate=0.1,
        epochs_per_eval=epochs_per_eval
    )

    # Format random search results
    random_results = {
        "method": "random_search",
        "num_samples": random_samples,
        "epochs_per_eval": epochs_per_eval,
        "best_fitness": random_fitness,
        "best_individual": random_best,
        "total_time": None,  # Will be filled from the saved file
        "fitness_scores": random_scores
    }

    # Format grid search results
    grid_results = {
        "method": "grid_search",
        "max_combinations": grid_max_combinations,
        "actual_combinations": len(grid_all),
        "epochs_per_eval": epochs_per_eval,
        "best_fitness": grid_fitness,
        "best_individual": grid_best,
        "total_time": None,  # Will be filled from the saved file
        "fitness_scores": grid_scores
    }

    # Load timing information from saved results
    try:
        with open("results/random_search_results.json", "r") as f:
            saved_random = json.load(f)
            random_results["total_time"] = saved_random["total_time"]

        with open("results/grid_search_results.json", "r") as f:
            saved_grid = json.load(f)
            grid_results["total_time"] = saved_grid["total_time"]
    except Exception as e:
        logger.error(f"Error loading timing information: {e}")

    # Create comparison plots
    plot_comparison_results(ga_results, random_results, grid_results)

    # Print comparison summary
    logger.info("\n" + "=" * 50)
    logger.info("Hyperparameter Tuning Methods Comparison Summary")
    logger.info("=" * 50)
    logger.info(
        f"Genetic Algorithm: Best Fitness = {ga_results['best_fitness']:.4f}, Evaluations = {ga_total_evals}, Time = {ga_results['total_time']:.1f}s")
    logger.info(
        f"Random Search: Best Fitness = {random_fitness:.4f}, Evaluations = {random_samples}, Time = {random_results['total_time']:.1f}s")
    logger.info(
        f"Grid Search: Best Fitness = {grid_fitness:.4f}, Evaluations = {len(grid_all)}, Time = {grid_results['total_time']:.1f}s")

    logger.info("\nBest CNN Architecture Details:")
    logger.info("\nGenetic Algorithm Best Architecture:")
    logger.info(f"  - Number of conv layers: {ga_results['best_individual']['conv_layers']}")
    logger.info(f"  - Learning rate: {ga_results['best_individual']['learning_rate']}")

    logger.info("\nRandom Search Best Architecture:")
    logger.info(f"  - Number of conv layers: {random_best['conv_layers']}")
    logger.info(f"  - Learning rate: {random_best['learning_rate']}")

    logger.info("\nGrid Search Best Architecture:")
    logger.info(f"  - Number of conv layers: {grid_best['conv_layers']}")
    logger.info(f"  - Learning rate: {grid_best['learning_rate']}")

    logger.info("\nComparison completed! Results saved to 'results/' directory.")


if __name__ == "__main__":
    # Run comparison with balanced evaluation budgets
    run_comparison(
        ga_population_size=8,
        ga_num_generations=5,
        random_samples=40,
        grid_max_combinations=40,
        epochs_per_eval=3
    )
