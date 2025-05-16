import unittest
import random
import numpy as np
import tensorflow as tf
import time
from typing import Dict, Any, List, Tuple
from copy import deepcopy

# Import functions from our module
from genetic_algorithms_starter import (
    generate_individual,
    initialize_population,
    select_parents,
    crossover,
    mutate,
    evolve_population,
    HYPERPARAMETER_SPACE
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


class TestGeneticAlgorithmComponents(unittest.TestCase):
    """Test individual components of the genetic algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample individuals and populations for testing
        self.sample_individual = generate_individual()
        self.population_size = 10
        self.population = initialize_population(self.population_size)
        self.fitness_scores = [random.random() for _ in range(self.population_size)]

        # Set mutation and crossover rates
        self.mutation_rate = 0.1

    def test_generate_individual(self):
        """Test if generate_individual creates valid individuals."""
        individual = generate_individual()

        # Check basic structure
        self.assertIn('conv_layers', individual)
        self.assertIn('learning_rate', individual)

        # Check values are from the search space
        self.assertIn(individual['conv_layers'], HYPERPARAMETER_SPACE['conv_layers'])
        self.assertIn(individual['learning_rate'], HYPERPARAMETER_SPACE['learning_rates'])

        # Check layer parameters
        for i in range(individual['conv_layers']):
            self.assertIn(f'filters_{i}', individual)
            self.assertIn(f'kernel_size_{i}', individual)
            self.assertIn(f'activation_{i}', individual)
            self.assertIn(f'pool_type_{i}', individual)
            self.assertIn(f'dropout_{i}', individual)

            self.assertIn(individual[f'filters_{i}'], HYPERPARAMETER_SPACE['filters'])
            self.assertIn(individual[f'kernel_size_{i}'], HYPERPARAMETER_SPACE['kernel_sizes'])
            self.assertIn(individual[f'activation_{i}'], HYPERPARAMETER_SPACE['activation_functions'])
            self.assertIn(individual[f'pool_type_{i}'], HYPERPARAMETER_SPACE['pool_types'])
            self.assertIn(individual[f'dropout_{i}'], HYPERPARAMETER_SPACE['dropout_rates'])

    def test_initialize_population(self):
        """Test if initialize_population creates a population of valid individuals."""
        population = initialize_population(self.population_size)

        # Check population size
        self.assertEqual(len(population), self.population_size)

        # Check each individual is valid
        for individual in population:
            self.assertIn('conv_layers', individual)
            self.assertIn('learning_rate', individual)

            for i in range(individual['conv_layers']):
                self.assertIn(f'filters_{i}', individual)
                self.assertIn(f'kernel_size_{i}', individual)
                self.assertIn(f'activation_{i}', individual)
                self.assertIn(f'pool_type_{i}', individual)
                self.assertIn(f'dropout_{i}', individual)

    def test_select_parents(self):
        """Test if select_parents selects the correct number of parents."""
        num_parents = 5
        parents = select_parents(self.population, self.fitness_scores, num_parents)

        # Check the number of parents
        self.assertEqual(len(parents), num_parents)

        # Check parents are from the population
        for parent in parents:
            self.assertIn(parent, self.population)

    def test_select_parents_tournament_pressure(self):
        """Test if tournament selection favors individuals with higher fitness."""
        # Create a population with controlled fitness scores
        population_size = 100
        population = initialize_population(population_size)

        # Assign linearly increasing fitness scores
        fitness_scores = [i / population_size for i in range(population_size)]

        # Run multiple selection trials
        num_trials = 100
        num_parents = 10
        selection_counts = [0] * population_size

        for _ in range(num_trials):
            parents = select_parents(population, fitness_scores, num_parents)
            for parent in parents:
                parent_idx = population.index(parent)
                selection_counts[parent_idx] += 1

        # Check if higher fitness individuals were selected more frequently
        # by comparing first and last quartiles
        quartile_size = population_size // 4
        first_quartile_count = sum(selection_counts[:quartile_size])
        last_quartile_count = sum(selection_counts[-quartile_size:])

        self.assertLess(first_quartile_count, last_quartile_count,
                        "Selection pressure test failed: lower fitness individuals were selected more often")

    def test_crossover(self):
        """Test if crossover produces valid offspring."""
        # Use two different individuals to ensure crossover has an effect
        parent1 = self.population[0]

        # Find a parent with different number of layers for better testing
        for p in self.population[1:]:
            if p['conv_layers'] != parent1['conv_layers']:
                parent2 = p
                break
        else:
            # If no different parent found, use the second one anyway
            parent2 = self.population[1]

        # Perform crossover
        offspring1, offspring2 = crossover(parent1, parent2)

        # Check offspring structure
        for offspring in [offspring1, offspring2]:
            self.assertIn('conv_layers', offspring)
            self.assertIn('learning_rate', offspring)

            for i in range(offspring['conv_layers']):
                self.assertIn(f'filters_{i}', offspring)
                self.assertIn(f'kernel_size_{i}', offspring)
                self.assertIn(f'activation_{i}', offspring)
                self.assertIn(f'pool_type_{i}', offspring)
                self.assertIn(f'dropout_{i}', offspring)

        # Verify offspring are different from parents
        # At least one should differ in some way due to crossover
        offspring_different = False
        for param in ['learning_rate', 'conv_layers']:
            if (offspring1[param] != parent1[param] or
                    offspring1[param] != parent2[param] or
                    offspring2[param] != parent1[param] or
                    offspring2[param] != parent2[param]):
                offspring_different = True
                break

        if not offspring_different:
            # Check layer parameters
            min_layers = min(parent1['conv_layers'], parent2['conv_layers'])
            for i in range(min_layers):
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    param_name = f'{param}_{i}'
                    if (offspring1[param_name] != parent1[param_name] or
                            offspring1[param_name] != parent2[param_name] or
                            offspring2[param_name] != parent1[param_name] or
                            offspring2[param_name] != parent2[param_name]):
                        offspring_different = True
                        break
                if offspring_different:
                    break

        self.assertTrue(offspring_different, "Crossover did not create different offspring")

    def test_mutate(self):
        """Test if mutation produces valid individuals."""
        # Clone the individual to ensure the original is not modified
        original = deepcopy(self.sample_individual)

        # Use a high mutation rate to ensure changes
        high_mutation_rate = 0.8
        mutated = mutate(original, high_mutation_rate)

        # Check mutated structure
        self.assertIn('conv_layers', mutated)
        self.assertIn('learning_rate', mutated)

        for i in range(mutated['conv_layers']):
            self.assertIn(f'filters_{i}', mutated)
            self.assertIn(f'kernel_size_{i}', mutated)
            self.assertIn(f'activation_{i}', mutated)
            self.assertIn(f'pool_type_{i}', mutated)
            self.assertIn(f'dropout_{i}', mutated)

        # Verify the original was not modified
        self.assertEqual(original['conv_layers'], self.sample_individual['conv_layers'])
        self.assertEqual(original['learning_rate'], self.sample_individual['learning_rate'])

        # Check if mutation had an effect (with high rate, something should change)
        something_changed = False

        if mutated['conv_layers'] != original['conv_layers'] or mutated['learning_rate'] != original['learning_rate']:
            something_changed = True
        else:
            # Check layer parameters
            min_layers = min(original['conv_layers'], mutated['conv_layers'])
            for i in range(min_layers):
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    param_name = f'{param}_{i}'
                    if mutated[param_name] != original[param_name]:
                        something_changed = True
                        break
                if something_changed:
                    break

        self.assertTrue(something_changed, "Mutation did not change the individual")

    def test_evolve_population(self):
        """Test if evolve_population produces a valid new generation."""
        next_generation = evolve_population(
            self.population, self.fitness_scores, self.population_size, self.mutation_rate
        )

        # Check population size
        self.assertEqual(len(next_generation), self.population_size)

        # Check each individual is valid
        for individual in next_generation:
            self.assertIn('conv_layers', individual)
            self.assertIn('learning_rate', individual)

            for i in range(individual['conv_layers']):
                self.assertIn(f'filters_{i}', individual)
                self.assertIn(f'kernel_size_{i}', individual)
                self.assertIn(f'activation_{i}', individual)
                self.assertIn(f'pool_type_{i}', individual)
                self.assertIn(f'dropout_{i}', individual)


class TestGeneticAlgorithmIntegration(unittest.TestCase):
    """Test integration of genetic algorithm components."""

    def setUp(self):
        """Set up test fixtures."""
        self.population_size = 10
        self.num_generations = 3
        self.mutation_rate = 0.1

    def test_evolution_over_generations(self):
        """Test evolution over multiple generations."""
        # Initialize population
        population = initialize_population(self.population_size)

        # Create mock fitness function (using number of layers as fitness for testing)
        def mock_fitness(individual):
            return individual['conv_layers'] / 5.0  # Normalize to [0-1]

        # Track the best individual and fitness history
        best_individual = None
        best_fitness = 0
        generation_stats = []

        # Evolution loop
        for generation in range(self.num_generations):
            # Calculate fitness for each individual
            fitness_scores = [mock_fitness(ind) for ind in population]

            # Find the best individual
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            gen_best_individual = population[gen_best_idx]

            # Update overall best
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = deepcopy(gen_best_individual)

            # Record statistics
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            generation_stats.append(avg_fitness)

            # Evolve population
            population = evolve_population(
                population, fitness_scores, self.population_size, self.mutation_rate
            )

        # Verify best individual is valid
        self.assertIsNotNone(best_individual)
        self.assertIn('conv_layers', best_individual)
        self.assertIn('learning_rate', best_individual)

        # Check evolution produces valid populations
        for individual in population:
            self.assertIn('conv_layers', individual)
            self.assertIn('learning_rate', individual)

            for i in range(individual['conv_layers']):
                self.assertIn(f'filters_{i}', individual)
                self.assertIn(f'kernel_size_{i}', individual)
                self.assertIn(f'activation_{i}', individual)
                self.assertIn(f'pool_type_{i}', individual)
                self.assertIn(f'dropout_{i}', individual)

    def test_performance(self):
        """Test performance of genetic algorithm operations."""
        # Initialize population
        start_time = time.time()
        population = initialize_population(self.population_size)
        init_time = time.time() - start_time

        # Generate random fitness scores
        fitness_scores = [random.random() for _ in range(self.population_size)]

        # Measure selection time
        start_time = time.time()
        parents = select_parents(population, fitness_scores, self.population_size // 2)
        selection_time = time.time() - start_time

        # Measure crossover time
        start_time = time.time()
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                offspring1, offspring2 = crossover(parents[i], parents[i + 1])
        crossover_time = time.time() - start_time

        # Measure mutation time
        start_time = time.time()
        for individual in population:
            mutated = mutate(individual, self.mutation_rate)
        mutation_time = time.time() - start_time

        # Measure evolution time
        start_time = time.time()
        next_generation = evolve_population(
            population, fitness_scores, self.population_size, self.mutation_rate
        )
        evolution_time = time.time() - start_time

        # Print performance results
        print("\nPerformance Benchmarks:")
        print(f"  Population Initialization ({self.population_size} individuals): {init_time:.6f} seconds")
        print(f"  Parent Selection: {selection_time:.6f} seconds")
        print(f"  Crossover: {crossover_time:.6f} seconds")
        print(f"  Mutation: {mutation_time:.6f} seconds")
        print(f"  Full Population Evolution: {evolution_time:.6f} seconds")

        # No assertions here, just performance reporting


class TestHyperparameterSearchSpace(unittest.TestCase):
    """Test the hyperparameter search space coverage."""

    def test_search_space_coverage(self):
        """Test if the genetic algorithm explores the search space effectively."""
        # Generate a large population to check search space coverage
        population_size = 100
        population = initialize_population(population_size)

        # Track which values appear for each hyperparameter
        hyperparameter_values = {
            'conv_layers': set(),
            'learning_rate': set()
        }

        layer_param_values = {
            'filters': set(),
            'kernel_size': set(),
            'activation': set(),
            'pool_type': set(),
            'dropout': set()
        }

        # Collect values from population
        for individual in population:
            hyperparameter_values['conv_layers'].add(individual['conv_layers'])
            hyperparameter_values['learning_rate'].add(individual['learning_rate'])

            for i in range(individual['conv_layers']):
                layer_param_values['filters'].add(individual[f'filters_{i}'])
                layer_param_values['kernel_size'].add(individual[f'kernel_size_{i}'])
                layer_param_values['activation'].add(individual[f'activation_{i}'])
                layer_param_values['pool_type'].add(individual[f'pool_type_{i}'])
                layer_param_values['dropout'].add(individual[f'dropout_{i}'])

        # Mapping from parameter names to search space keys
        param_to_key = {
            'conv_layers': 'conv_layers',
            'learning_rate': 'learning_rates',
            'filters': 'filters',
            'kernel_size': 'kernel_sizes',
            'activation': 'activation_functions',
            'pool_type': 'pool_types',
            'dropout': 'dropout_rates'
        }

        # Check if all values in the search space appear in the population
        for param, values in hyperparameter_values.items():
            search_space_key = param_to_key[param]
            self.assertTrue(values.issubset(set(HYPERPARAMETER_SPACE[search_space_key])))

            # With a large population, we should see most values
            coverage_pct = len(values) / len(HYPERPARAMETER_SPACE[search_space_key]) * 100
            print(
                f"{param} coverage: {coverage_pct:.1f}% ({len(values)}/{len(HYPERPARAMETER_SPACE[search_space_key])} values)")

        # Check layer parameter coverage
        for param, values in layer_param_values.items():
            search_space_key = param_to_key[param]
            self.assertTrue(values.issubset(set(HYPERPARAMETER_SPACE[search_space_key])))

            coverage_pct = len(values) / len(HYPERPARAMETER_SPACE[search_space_key]) * 100
            print(
                f"{param} coverage: {coverage_pct:.1f}% ({len(values)}/{len(HYPERPARAMETER_SPACE[search_space_key])} values)")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
