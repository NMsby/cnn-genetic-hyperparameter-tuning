import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
from typing import Dict, Any, List, Tuple
import os
import logging

# Import our genetic algorithm components
from genetic_algorithms_starter import (
    generate_individual,
    initialize_population,
    evaluate_fitness,
    select_parents,
    crossover,
    mutate,
    evolve_population,
    build_model,
    print_best_individual,
    plot_fitness_history,
    load_data
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
        logging.FileHandler("test_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestRun")


def mock_evaluate_fitness(
        individual: Dict[str, Any],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 1,
        batch_size: int = 64
) -> float:
    """
    A simplified version of evaluate_fitness that trains for fewer steps.
    This speeds up the testing process by limiting the training duration.

    Args:
        individual: The hyperparameters to evaluate
        x_train: Training data features
        y_train: Training data labels
        x_val: Validation data features
        y_val: Validation data labels
        epochs: Number of training epochs (kept low for testing)
        batch_size: Batch size for training

    Returns:
        float: Validation accuracy (fitness score)
    """
    # Build model
    model = build_model(individual)

    # Train for just a few steps to verify functionality
    history = model.fit(
        x_train[:1000],  # Use a small subset of training data
        y_train[:1000],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val[:500], y_val[:500]),  # Small subset of validation data
        verbose=0  # Silent training
    )

    # Return validation accuracy as fitness
    return history.history['val_accuracy'][-1]


def run_small_scale_test(
        population_size: int = 5,
        num_generations: int = 3,
        mutation_rate: float = 0.2,
        epochs_per_eval: int = 1
) -> Tuple[Dict[str, Any], float, List[float]]:
    """
    Run a small-scale test of the genetic algorithm.

    Args:
        population_size: Size of the population
        num_generations: Number of generations to evolve
        mutation_rate: Probability of mutation
        epochs_per_eval: Training epochs per fitness evaluation

    Returns:
        Tuple containing:
        - Best individual (hyperparameters)
        - Best fitness (validation accuracy)
        - Fitness history (list of best fitness values in each generation)
    """
    logger.info(f"Starting small-scale test with population={population_size}, generations={num_generations}")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load data
    logger.info("Loading data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    logger.info(
        f"Data loaded: {x_train.shape[0]} training, {x_val.shape[0]} validation, {x_test.shape[0]} test samples")

    # Initialize population
    logger.info("Initializing population...")
    population = initialize_population(population_size)

    # Track the best individual and fitness history
    best_individual = None
    best_fitness = 0
    fitness_history = []
    avg_fitness_history = []

    # Main evolution loop
    for generation in range(num_generations):
        generation_start_time = time.time()
        logger.info(f"Generation {generation + 1}/{num_generations}")

        # Evaluate fitness for each individual
        fitness_scores = []
        for i, individual in enumerate(population):
            logger.info(f"  Evaluating individual {i + 1}/{population_size}...")

            # Use the simplified fitness evaluation for testing
            fitness = mock_evaluate_fitness(
                individual, x_train, y_train, x_val, y_val,
                epochs=epochs_per_eval
            )
            fitness_scores.append(fitness)
            logger.info(f"  Individual {i + 1} fitness: {fitness:.4f}")

        # Find the best individual in this generation
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_individual = population[gen_best_idx]

        # Calculate average fitness
        gen_avg_fitness = sum(fitness_scores) / len(fitness_scores)
        avg_fitness_history.append(gen_avg_fitness)

        # Update overall best if better
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual.copy()
            logger.info(f"  New best individual found! Fitness: {best_fitness:.4f}")

        # Add to fitness history
        fitness_history.append(gen_best_fitness)

        # Print generation stats
        generation_time = time.time() - generation_start_time
        logger.info(f"  Best fitness in generation: {gen_best_fitness:.4f}")
        logger.info(f"  Average fitness in generation: {gen_avg_fitness:.4f}")
        logger.info(f"  Overall best fitness: {best_fitness:.4f}")
        logger.info(f"  Generation time: {generation_time:.2f} seconds")

        # If we've reached the last generation, we're done
        if generation == num_generations - 1:
            break

        # Evolve population
        logger.info("  Evolving population...")
        population = evolve_population(population, fitness_scores, population_size, mutation_rate)

    # Final evaluation
    logger.info("\nTest Run Complete!")
    logger.info(f"Best fitness achieved: {best_fitness:.4f}")
    logger.info("Best architecture:")
    print_best_individual(best_individual, best_fitness)

    # Plot fitness history
    plt.figure(figsize=(10, 6))

    # Plot best fitness
    plt.plot(range(1, num_generations + 1), fitness_history, 'b-', marker='o', label='Best Fitness')

    # Plot average fitness
    plt.plot(range(1, num_generations + 1), avg_fitness_history, 'r-', marker='x', label='Average Fitness')

    plt.xlabel('Generation')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.title('Fitness History - Small Scale Test')
    plt.xticks(range(1, num_generations + 1))
    plt.grid(True)
    plt.legend()

    # Save plot to file
    plt.savefig("results/small_scale_test_fitness.png")
    logger.info("Fitness plot saved to results/small_scale_test_fitness.png")

    # Show plot
    plt.show()

    return best_individual, best_fitness, fitness_history


def test_best_model(best_individual: Dict[str, Any]) -> float:
    """
    Test the best model on a small subset of the test data.

    Args:
        best_individual: The best hyperparameters found

    Returns:
        float: Test accuracy
    """
    logger.info("\nTesting best model on a subset of test data...")

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Build model with best hyperparameters
    model = build_model(best_individual)

    # Train on combined training and validation data
    # Use smaller subset and fewer epochs for testing
    x_train_small = np.concatenate([x_train[:1000], x_val[:500]])
    y_train_small = np.concatenate([y_train[:1000], y_val[:500]])

    model.fit(
        x_train_small, y_train_small,
        epochs=2,  # Very few epochs for testing
        batch_size=64,
        verbose=1
    )

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(x_test[:500], y_test[:500], verbose=1)
    logger.info(f"Test accuracy (subset): {test_accuracy:.4f}")

    return test_accuracy


if __name__ == "__main__":
    logger.info("Starting small-scale genetic algorithm test")

    try:
        # Run small-scale test
        best_individual, best_fitness, fitness_history = run_small_scale_test(
            population_size=5,
            num_generations=3,
            mutation_rate=0.2,
            epochs_per_eval=1
        )

        # Test best model
        test_accuracy = test_best_model(best_individual)

        logger.info("Small-scale test completed successfully")

    except Exception as e:
        logger.error(f"Error during small-scale test: {e}", exc_info=True)
        raise
