import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random
import time
import os
import json
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Define hyperparameter search space
HYPERPARAMETER_SPACE = {
    'conv_layers': [1, 2, 3, 4, 5],  # Number of convolutional layers
    'filters': [16, 32, 64, 128, 256],  # Number of filters per layer
    'kernel_sizes': [3, 5, 7],  # Kernel sizes for convolutional layers
    'pool_types': ['max', 'avg', 'none'],  # Pooling types
    'learning_rates': [0.1, 0.01, 0.001, 0.0001],  # Learning rates
    'activation_functions': ['relu', 'elu', 'leaky_relu'],  # Activation functions
    'dropout_rates': [0.0, 0.25, 0.5]  # Dropout rates
}

# Global cache for fitness evaluations
fitness_cache = {}


# Function to hash an individual for caching
def hash_individual(individual: Dict[str, Any]) -> str:
    """
    Create a hash of an individual for caching purposes.

    Args:
        individual: The individual to hash

    Returns:
        str: A hash string representing the individual
    """
    # Convert the individual to a sorted, deterministic string representation
    sorted_items = sorted(individual.items())
    individual_str = json.dumps(sorted_items)

    # Create a hash
    return hashlib.md5(individual_str.encode()).hexdigest()


def load_data():
    """
    Load and preprocess CIFAR-10 dataset with optimized memory usage.
    """
    global _train_data, _val_data, _test_data

    # Check if data is already loaded in global variables
    if '_train_data' in globals() and '_val_data' in globals() and '_test_data' in globals():
        return _train_data, _val_data, _test_data

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create a validation set
    val_size = 5000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    # Store in global variables to avoid reloading
    _train_data = (x_train, y_train)
    _val_data = (x_val, y_val)
    _test_data = (x_test, y_test)

    return _train_data, _val_data, _test_data


def build_model(hyperparameters: Dict[str, Any]) -> tf.keras.Model:
    """
    Build a CNN model based on provided hyperparameters with optimized configuration.
    """
    # Use mixed precision for faster training if available
    if tf.config.list_physical_devices('GPU'):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model = Sequential()

    # Add convolutional layers
    for i in range(hyperparameters['conv_layers']):
        # Add convolutional layer
        if i == 0:
            # First layer needs input shape
            model.add(Conv2D(
                filters=hyperparameters[f'filters_{i}'],
                kernel_size=hyperparameters[f'kernel_size_{i}'],
                activation=hyperparameters[f'activation_{i}'],
                padding='same',
                input_shape=(32, 32, 3),
                use_bias=False  # Reduce parameters
            ))
        else:
            model.add(Conv2D(
                filters=hyperparameters[f'filters_{i}'],
                kernel_size=hyperparameters[f'kernel_size_{i}'],
                activation=hyperparameters[f'activation_{i}'],
                padding='same',
                use_bias=False  # Reduce parameters
            ))

        # Add pooling layer if specified
        if hyperparameters[f'pool_type_{i}'] == 'max':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif hyperparameters[f'pool_type_{i}'] == 'avg':
            model.add(AveragePooling2D(pool_size=(2, 2)))

        # Add dropout if rate > 0
        if hyperparameters[f'dropout_{i}'] > 0:
            model.add(Dropout(hyperparameters[f'dropout_{i}']))

    # Add flatten layer
    model.add(Flatten())

    # Add dense layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Add output layer
    model.add(Dense(10, activation='softmax'))

    # Compile model with optimized configuration
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hyperparameters['learning_rate'],
        epsilon=1e-7,  # Higher epsilon for better numerical stability
        amsgrad=True  # Use AMSGrad variant
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def evaluate_fitness_worker(worker_args):
    """
    Worker function for parallel fitness evaluation.

    Args:
        worker_args: Tuple containing (individual, epochs, batch_size, x_train_subset, y_train_subset, x_val, y_val)

    Returns:
        Tuple of (individual_hash, fitness)
    """
    individual, epochs, batch_size, x_train, y_train, x_val, y_val = worker_args

    # Generate a hash for this individual
    individual_hash = hash_individual(individual)

    # Check if we've already evaluated this individual
    global fitness_cache
    if individual_hash in fitness_cache:
        return individual_hash, fitness_cache[individual_hash]

    # Otherwise, evaluate the individual
    model = build_model(individual)

    # Add early stopping to save time
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=1,
        restore_best_weights=True,
        min_delta=0.005  # Only stop if improving by less than 0.5%
    )

    # Use a smaller batch for initial epochs (faster convergence) and larger for later epochs (more stable)
    initial_epochs = min(2, epochs)

    # Train for initial epochs with smaller batch size
    if initial_epochs > 0:
        model.fit(
            x_train, y_train,
            epochs=initial_epochs,
            batch_size=batch_size // 2,  # Smaller batch size
            validation_data=(x_val, y_val),
            verbose=0,
            callbacks=[early_stopping]
        )

    # Train for remaining epochs with larger batch size
    history = model.fit(
        x_train, y_train,
        initial_epoch=initial_epochs,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=0,
        callbacks=[early_stopping]
    )

    # Get the validation accuracy
    fitness = history.history['val_accuracy'][-1]

    # Cache the result
    fitness_cache[individual_hash] = fitness

    # Clean up to reduce memory usage
    tf.keras.backend.clear_session()

    return individual_hash, fitness


def evaluate_fitness_parallel(
        population: List[Dict[str, Any]],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 5,
        batch_size: int = 64,
        max_workers: Optional[int] = None
) -> List[float]:
    """
    Evaluate fitness for a population in parallel.

    Args:
        population: List of individuals to evaluate
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        max_workers: Maximum number of parallel workers

    Returns:
        List of fitness scores
    """
    # Determine the number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(population))

    # Prepare worker arguments
    worker_args = [
        (individual, epochs, batch_size, x_train, y_train, x_val, y_val)
        for individual in population
    ]

    # Run evaluations in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(evaluate_fitness_worker, worker_args))

    # Sort results by their original order in the population
    individual_hashes = [hash_individual(ind) for ind in population]
    fitness_scores = []

    for ind_hash in individual_hashes:
        for result_hash, fitness in results:
            if result_hash == ind_hash:
                fitness_scores.append(fitness)
                break

    return fitness_scores


def generate_individual() -> Dict[str, Any]:
    """
    Generate a random individual (set of hyperparameters).
    """
    # Start with basic hyperparameters
    individual = {
        'conv_layers': random.choice(HYPERPARAMETER_SPACE['conv_layers']),
        'learning_rate': random.choice(HYPERPARAMETER_SPACE['learning_rates'])
    }

    # For each conv layer, add specific hyperparameters
    for i in range(individual['conv_layers']):
        individual[f'filters_{i}'] = random.choice(HYPERPARAMETER_SPACE['filters'])
        individual[f'kernel_size_{i}'] = random.choice(HYPERPARAMETER_SPACE['kernel_sizes'])
        individual[f'activation_{i}'] = random.choice(HYPERPARAMETER_SPACE['activation_functions'])
        individual[f'pool_type_{i}'] = random.choice(HYPERPARAMETER_SPACE['pool_types'])
        individual[f'dropout_{i}'] = random.choice(HYPERPARAMETER_SPACE['dropout_rates'])

    return individual


def initialize_population(population_size: int) -> List[Dict[str, Any]]:
    """
    Initialize a population of random individuals.
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
    """
    parents = []

    for _ in range(num_parents):
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        # Select the best individual from the tournament
        winner_relative_idx = np.argmax(tournament_fitness)
        winner_idx = tournament_indices[winner_relative_idx]

        parents.append(population[winner_idx])

    return parents


def crossover(
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform crossover between two parents to create two offspring.
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
        mutation_rate: float = 0.1,
        elitism_count: int = 1
) -> List[Dict[str, Any]]:
    """
    Evolve the population to the next generation with optimized selection.
    """
    # Create a new empty generation
    next_generation = []

    # Apply elitism: Keep the best individual(s)
    if population_size >= 10:
        elitism_count = 2  # Use more elites for larger populations

    # Find and add the elite individuals
    elite_indices = np.argsort(fitness_scores)[-elitism_count:]
    for idx in elite_indices:
        next_generation.append(population[idx].copy())

    # Select parents for producing offspring
    num_parents_needed = max(population_size - elitism_count, 2)
    parents = select_parents(population, fitness_scores, num_parents_needed)

    # Create offspring through crossover and mutation until we reach population_size
    offspring_needed = population_size - len(next_generation)
    offspring_pairs_needed = (offspring_needed + 1) // 2  # Ceiling division

    for _ in range(offspring_pairs_needed):
        # Select two parents for crossover
        parent1, parent2 = random.sample(parents, 2)

        # Create offspring through crossover
        offspring1, offspring2 = crossover(parent1, parent2)

        # Apply mutation to offspring
        offspring1 = mutate(offspring1, mutation_rate)
        offspring2 = mutate(offspring2, mutation_rate)

        # Add offspring to the next generation
        next_generation.append(offspring1)
        if len(next_generation) < population_size:  # Check if we need the second offspring
            next_generation.append(offspring2)

    # Ensure the correct population size (should be guaranteed by the logic above)
    assert len(
        next_generation) == population_size, f"Expected population size {population_size}, got {len(next_generation)}"

    return next_generation


def run_genetic_algorithm(
        population_size: int = 10,
        num_generations: int = 10,
        mutation_rate: float = 0.1,
        epochs_per_eval: int = 5,
        batch_size: int = 64,
        max_workers: Optional[int] = None,
        save_results: bool = True
) -> Tuple[Dict[str, Any], float, List[float]]:
    """
    Run the genetic algorithm to optimize CNN hyperparameters.

    Args:
        population_size: Number of individuals in each generation
        num_generations: Number of generations to evolve
        mutation_rate: Probability of mutation for each hyperparameter
        epochs_per_eval: Number of training epochs for each fitness evaluation
        batch_size: Batch size for training
        max_workers: Maximum number of parallel workers
        save_results: Whether to save results to disk

    Returns:
        Tuple containing:
        - Best individual (hyperparameters)
        - Best fitness (validation accuracy)
        - Fitness history (list of best fitness values in each generation)
    """
    print(f"Starting genetic algorithm with {population_size} individuals and {num_generations} generations")

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Initialize population
    population = initialize_population(population_size)

    # Track the best individual and fitness history
    best_individual = None
    best_fitness = 0
    fitness_history = []
    population_history = []
    fitness_scores_history = []

    # Main loop
    total_start_time = time.time()
    for generation in range(num_generations):
        generation_start_time = time.time()
        print(f"\nGeneration {generation + 1}/{num_generations}")

        # Evaluate fitness for each individual in parallel
        fitness_scores = evaluate_fitness_parallel(
            population, x_train, y_train, x_val, y_val,
            epochs=epochs_per_eval, batch_size=batch_size, max_workers=max_workers
        )

        # Save population and fitness scores for analysis
        population_history.append([ind.copy() for ind in population])
        fitness_scores_history.append(fitness_scores.copy())

        # Find the best individual in this generation
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_individual = population[gen_best_idx]

        # Update overall best if better
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual.copy()
            print(f"  New best individual found! Fitness: {best_fitness:.4f}")

        # Add to fitness history
        fitness_history.append(gen_best_fitness)

        # Print generation stats
        generation_time = time.time() - generation_start_time
        print(f"  Best fitness in generation: {gen_best_fitness:.4f}")
        print(f"  Average fitness in generation: {np.mean(fitness_scores):.4f}")
        print(f"  Generation time: {generation_time:.2f} seconds")

        # If we've reached the last generation, we're done
        if generation == num_generations - 1:
            break

        # Evolve population
        population = evolve_population(population, fitness_scores, population_size, mutation_rate)

    # Print final stats
    total_time = time.time() - total_start_time
    print(f"\nGenetic algorithm completed in {total_time:.2f} seconds")
    print(f"Best fitness achieved: {best_fitness:.4f}")

    # Save results if requested
    if save_results:
        results = {
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "fitness_history": fitness_history,
            "parameters": {
                "population_size": population_size,
                "num_generations": num_generations,
                "mutation_rate": mutation_rate,
                "epochs_per_eval": epochs_per_eval,
                "batch_size": batch_size
            },
            "total_time": total_time
        }

        with open("results/optimized_ga_results.json", "w") as f:
            # Convert NumPy types to native Python types for JSON serialization
            json_results = {
                k: (float(v) if isinstance(v, (np.float32, np.float64)) else
                    [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v] if isinstance(v, list) else
                    v)
                for k, v in results.items()
            }
            json.dump(json_results, f, indent=2)

        print(f"Results saved to results/optimized_ga_results.json")

    return best_individual, best_fitness, fitness_history


def print_best_individual(individual: Dict[str, Any], fitness: float) -> None:
    """
    Print the hyperparameters and fitness of the best individual.
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


def plot_fitness_history(fitness_history: List[float]) -> None:
    """
    Plot the fitness history of the best individual in each generation.
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

    plt.savefig("results/optimized_fitness_history.png", dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_model(individual: Dict[str, Any], epochs: int = 10) -> float:
    """
    Evaluate a model with given hyperparameters on the test set.

    Args:
        individual: Hyperparameters for the model
        epochs: Number of training epochs

    Returns:
        float: Test accuracy
    """
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Combine training and validation data
    x_train_full = np.concatenate([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Build model
    model = build_model(individual)

    # Add callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=1e-6
        )
    ]

    # Train model
    model.fit(
        x_train_full, y_train_full,
        epochs=epochs,
        batch_size=128,  # Larger batch size for full training
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

    return test_accuracy


def benchmark():
    """
    Run a benchmark to compare optimized and original implementation.
    """
    print("Running benchmark...")

    # Parameters for the benchmark
    population_size = 5
    num_generations = 2
    mutation_rate = 0.1
    epochs_per_eval = 2

    # Import original implementation (assuming it's in genetic_algorithms_starter.py)
    import genetic_algorithms_starter as gas

    # Run original implementation
    print("\nRunning original implementation...")
    start_time = time.time()
    _, _, _ = gas.run_genetic_algorithm(
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        epochs_per_eval=epochs_per_eval
    )
    original_time = time.time() - start_time

    # Run optimized implementation
    print("\nRunning optimized implementation...")
    start_time = time.time()
    _, _, _ = run_genetic_algorithm(
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        epochs_per_eval=epochs_per_eval,
        save_results=False
    )
    optimized_time = time.time() - start_time

    # Print results
    print("\nBenchmark Results:")
    print(f"Original implementation: {original_time:.2f} seconds")
    print(f"Optimized implementation: {optimized_time:.2f} seconds")
    print(f"Speedup: {original_time / optimized_time:.2f}x")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized Genetic Algorithm for CNN Hyperparameter Tuning")
    parser.add_argument("--population", type=int, default=10, help="Population size")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per fitness evaluation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark comparison")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate best model on test set")
    parser.add_argument("--no_save", action="store_true", help="Don't save results to disk")

    args = parser.parse_args()

    if args.benchmark:
        benchmark()
    else:
        # Run genetic algorithm
        best_individual, best_fitness, fitness_history = run_genetic_algorithm(
            population_size=args.population,
            num_generations=args.generations,
            mutation_rate=args.mutation_rate,
            epochs_per_eval=args.epochs,
            batch_size=args.batch_size,
            max_workers=args.workers,
            save_results=not args.no_save
        )

        # Print results
        print("\nGenetic Algorithm Results:")
        print_best_individual(best_individual, best_fitness)

        # Plot fitness history
        plot_fitness_history(fitness_history)

        # Evaluate on the test set if requested
        if args.evaluate:
            print("\nEvaluating best model on test set...")
            test_accuracy = evaluate_model(best_individual, epochs=10)
            print(f"Test accuracy: {test_accuracy:.4f}")
