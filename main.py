#!/usr/bin/env python3
"""
Main entry point for CNN hyperparameter tuning with genetic algorithms.

This script provides a command-line interface to run the genetic algorithm,
baseline methods, and visualization tools.
"""

import argparse
import os
import time
import tensorflow as tf
import numpy as np
import random
from pathlib import Path

# Create results directory if it doesn't exist
os.makedirs("results/figures", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def main():
    """Main entry point for the application."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="CNN Hyperparameter Tuning with Genetic Algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Genetic Algorithm parser
    ga_parser = subparsers.add_parser("ga", help="Run genetic algorithm")
    ga_parser.add_argument("--population", type=int, default=10, help="Population size")
    ga_parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    ga_parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
    ga_parser.add_argument("--epochs", type=int, default=5, help="Epochs per fitness evaluation")
    ga_parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    ga_parser.add_argument("--evaluate", action="store_true", help="Evaluate best model on test set")

    # Optimized algorithm parser
    opt_parser = subparsers.add_parser("optimized", help="Run optimized genetic algorithm")
    opt_parser.add_argument("--population", type=int, default=10, help="Population size")
    opt_parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    opt_parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
    opt_parser.add_argument("--epochs", type=int, default=5, help="Epochs per fitness evaluation")
    opt_parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    opt_parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    opt_parser.add_argument("--evaluate", action="store_true", help="Evaluate best model on test set")

    # Baseline parser
    baseline_parser = subparsers.add_parser("baseline", help="Run baseline methods")
    baseline_parser.add_argument("--method", choices=["random", "grid", "all"], default="all",
                                 help="Baseline method to run")
    baseline_parser.add_argument("--samples", type=int, default=20,
                                 help="Number of samples for random/grid search")
    baseline_parser.add_argument("--epochs", type=int, default=3, help="Epochs per evaluation")

    # Ablation parser
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation studies")
    ablation_parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    ablation_parser.add_argument("--focused", action="store_true", help="Run focused ablation study")
    ablation_parser.add_argument("--visualize", action="store_true", help="Visualize ablation results")

    # Visualization parser (continued)
    viz_parser = subparsers.add_parser("visualize", help="Run visualization tools")
    viz_parser.add_argument("--type", choices=["architecture", "features", "results", "all"],
                            default="all", help="Type of visualization to run")
    viz_parser.add_argument("--model_path", type=str,
                            help="Path to model results JSON file for visualization")

    # Benchmark parser
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("--compare", action="store_true",
                                  help="Compare optimized vs original implementation")

    # Parse arguments
    args = parser.parse_args()

    # Handle command
    if args.command == "ga":
        run_genetic_algorithm(args)
    elif args.command == "optimized":
        run_optimized_algorithm(args)
    elif args.command == "baseline":
        run_baseline(args)
    elif args.command == "ablation":
        run_ablation(args)
    elif args.command == "visualize":
        run_visualization(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help()


def run_genetic_algorithm(args):
    """Run the standard genetic algorithm."""
    from src.core.genetic_algorithm import run_genetic_algorithm
    from src.core.utils import print_best_individual, plot_fitness_history
    from src.core.model_builder import load_data, build_model

    print(f"Running genetic algorithm with population={args.population}, generations={args.generations}")

    # Run genetic algorithm
    start_time = time.time()
    best_individual, best_fitness, fitness_history = run_genetic_algorithm(
        population_size=args.population,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        epochs_per_eval=args.epochs,
        batch_size=args.batch_size
    )
    total_time = time.time() - start_time

    # Print results
    print(f"\nGenetic algorithm completed in {total_time:.2f} seconds")
    print_best_individual(best_individual, best_fitness)

    # Plot fitness history
    plot_fitness_history(fitness_history, "results/figures/ga_fitness_history.png")

    # Save results
    save_results(best_individual, best_fitness, fitness_history, total_time, "results/ga_results.json")

    # Evaluate on the test set if requested
    if args.evaluate:
        print("\nEvaluating best model on test set...")
        evaluate_model(best_individual)


def run_optimized_algorithm(args):
    """Run the optimized genetic algorithm."""
    from src.experiments.optimized_algorithm import (
        run_genetic_algorithm as run_optimized,
        evaluate_model
    )

    print(f"Running optimized genetic algorithm with population={args.population}, generations={args.generations}")

    # Run optimized genetic algorithm
    best_individual, best_fitness, fitness_history = run_optimized(
        population_size=args.population,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        epochs_per_eval=args.epochs,
        batch_size=args.batch_size,
        max_workers=args.workers
    )

    # Evaluate on the test set if requested
    if args.evaluate:
        print("\nEvaluating best model on test set...")
        test_accuracy = evaluate_model(best_individual, epochs=10)
        print(f"Test accuracy: {test_accuracy:.4f}")


def run_baseline(args):
    """Run baseline methods."""
    from src.experiments.baseline_methods import run_random_search, run_grid_search, run_comparison

    if args.method == "random":
        print(f"Running random search with {args.samples} samples")
        run_random_search(num_samples=args.samples, epochs_per_eval=args.epochs)
    elif args.method == "grid":
        print(f"Running grid search with max {args.samples} combinations")
        run_grid_search(max_combinations=args.samples, epochs_per_eval=args.epochs)
    else:  # "all"
        print("Running comparison of all methods")
        run_comparison(
            ga_population_size=args.samples // 4,  # Adjust to get similar total evaluations
            ga_num_generations=4,
            random_samples=args.samples,
            grid_max_combinations=args.samples,
            epochs_per_eval=args.epochs
        )


def run_ablation(args):
    """Run ablation studies."""
    from src.experiments.ablation_studies import (
        run_multiple_ablation_studies,
        run_focused_ablation_study,
        visualize_ablation_results
    )

    if args.visualize:
        print("Visualizing existing ablation results")
        visualize_ablation_results()
    elif args.focused:
        print("Running focused ablation study")
        run_focused_ablation_study()
    else:
        print("Running multiple ablation studies")
        run_multiple_ablation_studies(parallel=args.parallel)


def run_visualization(args):
    """Run visualization tools."""
    from src.visualization.architecture_viz import visualize_architecture
    from src.visualization.results_viz import run_visualization_suite

    if args.type == "architecture":
        print("Visualizing CNN architecture")
        visualize_architecture()
    elif args.type == "features":
        print("Visualizing CNN features")
        # Requires a trained model to visualize
        if args.model_path:
            from src.visualization.architecture_viz import load_and_visualize_best_model
            load_and_visualize_best_model(args.model_path)
        else:
            print("Error: Model path required for feature visualization")
    elif args.type == "results":
        print("Visualizing results")
        run_visualization_suite()
    else:  # "all"
        print("Running all visualizations")
        visualize_architecture()
        run_visualization_suite()


def run_benchmark(args):
    """Run benchmarks."""
    from src.experiments.optimized_algorithm import benchmark

    if args.compare:
        print("Running comparison benchmark")
        benchmark()
    else:
        print("Running performance benchmark")
        # Default benchmark
        benchmark()


def save_results(best_individual, best_fitness, fitness_history, total_time, filename):
    """Save results to a JSON file."""
    import json

    results = {
        "best_individual": best_individual,
        "best_fitness": float(best_fitness),  # Convert numpy types to native Python
        "fitness_history": [float(x) for x in fitness_history],
        "total_time": float(total_time)
    }

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {filename}")


def evaluate_model(best_individual, epochs=10):
    """Evaluate the best model on the test set."""
    from src.core.model_builder import load_data, build_model

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Combine training and validation data
    x_train_full = np.concatenate([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Build model
    model = build_model(best_individual)

    # Train model
    model.fit(
        x_train_full, y_train_full,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")

    return test_accuracy


if __name__ == "__main__":
    main()
