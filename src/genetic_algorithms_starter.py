"""
Genetic Algorithm for CNN Hyperparameter Tuning
==============================================

This module implements a genetic algorithm framework for optimizing
hyperparameters of Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset.

The genetic algorithm optimizes various hyperparameters including:
- Number of convolutional layers
- Number of filters per layer
- Kernel sizes
- Pooling strategies
- Activation functions
- Dropout rates
- Learning rate

The implementation follows these main steps:
1. Generate an initial population of random CNN architectures
2. Evaluate the fitness of each architecture (validation accuracy)
3. Select the fittest individuals for reproduction
4. Create offspring through crossover and mutation
5. Replace the old population with the new generation
6. Repeat steps 2-5 for multiple generations

Usage:
    best_individual, best_fitness, fitness_history = run_genetic_algorithm(
        population_size=10,
        num_generations=10,
        mutation_rate=0.1,
        epochs_per_eval=3
    )
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import time
from typing import List, Dict, Tuple, Any, Optional

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


# Load and preprocess data
def load_data():
    """
    Load and preprocess CIFAR-10 dataset.

    The function performs the following preprocessing steps:
    1. Load the CIFAR-10 dataset using TensorFlow's API
    2. Normalize pixel values to range [0, 1]
    3. One-hot encode the class labels
    4. Split the training data to create a validation set

    Returns:
        tuple: Containing three pairs of (data, labels) for training, validation, and test sets
            - Training set: (x_train, y_train)
            - Validation set: (x_val, y_val)
            - Test set: (x_test, y_test)
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1,
    # This helps with training stability and convergence
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    # Converts class vectors (integers) to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create a validation set from the last 5000 samples of the training set
    # This will be used to evaluate the fitness of our models during evolution
    val_size = 5000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# Define hyperparameter search space
# These are the parameters the genetic algorithm will optimize
HYPERPARAMETER_SPACE = {
    'conv_layers': [1, 2, 3, 4, 5],                         # Number of convolutional layers
    'filters': [16, 32, 64, 128, 256],                      # Number of filters per layer
    'kernel_sizes': [3, 5, 7],                              # Kernel sizes for convolutional layers
    'pool_types': ['max', 'avg', 'none'],                   # Pooling types
    'learning_rates': [0.1, 0.01, 0.001, 0.0001],           # Learning rates
    'activation_functions': ['relu', 'elu', 'leaky_relu'],  # Activation functions
    'dropout_rates': [0.0, 0.25, 0.5]                       # Dropout rates
}


# Model builder function
def build_model(hyperparameters: Dict[str, Any]) -> tf.keras.Model:
    """
    Build a CNN model based on provided hyperparameters.

    This function creates a Sequential model with hyperparameter-determined:
    - Number of convolutional layers
    - Number of filters in each layer
    - Kernel size for each layer
    - Activation function for each layer
    - Pooling type for each layer
    - Dropout rate for each layer

    Args:
        hyperparameters (Dict[str, Any]): Dictionary containing hyperparameters for the model
            Must include:
            - 'conv_layers': Number of convolutional layers
            - 'learning_rate': Learning rate for the optimizer
            - For each layer i (0 to conv_layers-1):
                - f'filters_{i}': Number of filters
                - f'kernel_size_{i}': Kernel size
                - f'activation_{i}': Activation function
                - f'pool_type_{i}': Pooling type ('max', 'avg', or 'none')
                - f'dropout_{i}': Dropout rate

    Returns:
        tf.keras.Model: Compiled CNN model ready for training
    """
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
                input_shape=(32, 32, 3)     # CIFAR-10 images are 32x32x3
            ))
        else:
            model.add(Conv2D(
                filters=hyperparameters[f'filters_{i}'],
                kernel_size=hyperparameters[f'kernel_size_{i}'],
                activation=hyperparameters[f'activation_{i}'],
                padding='same'
            ))

        # Add pooling layer if specified
        # Different pooling strategies can affect feature extraction quality
        if hyperparameters[f'pool_type_{i}'] == 'max':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif hyperparameters[f'pool_type_{i}'] == 'avg':
            model.add(AveragePooling2D(pool_size=(2, 2)))

        # Add dropout if rate > 0
        # Dropout helps prevent overfitting by randomly setting inputs to zero
        if hyperparameters[f'dropout_{i}'] > 0:
            model.add(Dropout(hyperparameters[f'dropout_{i}']))

    # Add flatten layer to convert 3D feature maps to 1D feature vectors
    model.add(Flatten())

    # Add dense layer (fully connected)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))     # Fixed dropout for the dense layer

    # Add output layer with 10 neurons (one for each CIFAR-10 class)
    model.add(Dense(10, activation='softmax'))

    # Compile model with specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',    # Appropriate for one-hot encoded labels
        metrics=['accuracy']                # We'll use accuracy as our fitness measure
    )

    return model


# Function to flatten hyperparameters from the search space into model parameters
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
    # Example structure:
    individual = {
        # Randomly select the number of convolutional layers
        'conv_layers': random.choice(HYPERPARAMETER_SPACE['conv_layers']),
        # Randomly select the learning rate
        'learning_rate': random.choice(HYPERPARAMETER_SPACE['learning_rates'])
    }

    # For each convolutional layer, generate layer-specific hyperparameters
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
        population_size (int): The number of individuals to create

    Returns:
        List[Dict[str, Any]]: List of randomly generated individuals

    TODO: Complete the implementation to generate multiple random individuals.
    """
    # TODO: Student implementation
    return [generate_individual() for _ in range(population_size)]


def evaluate_fitness(
        individual: Dict[str, Any],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 5,
        batch_size: int = 64
) -> float:
    """
    Build and train a model with the given hyperparameters and return validation accuracy.

    This function serves as the fitness function for our genetic algorithm.
    Higher validation accuracy indicates a better individual.

    Process:
    1. Build model using the individual's hyperparameters
    2. Train the model on the training data
    3. Evaluate the model on validation data
    4. Return validation accuracy as fitness

    Args:
        individual (Dict[str, Any]): The hyperparameters to evaluate
        x_train (np.ndarray): Training data features
        y_train (np.ndarray): Training data labels
        x_val (np.ndarray): Validation data features
        y_val (np.ndarray): Validation data labels
        epochs (int, optional): Number of training epochs. Defaults to 5.
        batch_size (int, optional): Batch size for training. Defaults to 64.

    Returns:
        float: Validation accuracy (fitness score) in range [0, 1]

    TODO: Complete the implementation to evaluate an individual's fitness.
    """
    # TODO: Student implementation
    # Build model
    model = build_model(individual)

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=0   # Set to 1 for training progress, 0 for silent
    )

    # Return validation accuracy as fitness
    return history.history['val_accuracy'][-1]


def select_parents(
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        num_parents: int,
        tournament_size: int = 3
) -> List[Dict[str, Any]]:
    """
    Select individuals to be parents for the next generation using tournament selection.

    Tournament selection: randomly select k individuals and choose the best one.
    Repeat until we have num_parents parents.

    Args:
        population (List[Dict[str, Any]]): List of individuals
        fitness_scores (List[float]): Fitness scores corresponding to each individual
        num_parents (int): Number of parents to select
        tournament_size (int, optional): Number of individuals in each tournament.
                                         Larger values increase selection pressure.
                                         Defaults to 3.

    Returns:
        List[Dict[str, Any]]: Selected parents

    """
    parents = []

    # Continue selecting parents until we have enough
    for _ in range(num_parents):
        # Randomly select tournament_size individuals for this tournament
        # If tournament_size is larger than population size, limit it
        actual_tournament_size = min(tournament_size, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)

        # Get the fitness scores for the selected individuals
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        # Find the winner (the individual with the highest fitness in this tournament)
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
    # Create copies of parents to avoid modifying the originals
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Get min and max layer counts from parents
    min_layers = min(parent1['conv_layers'], parent2['conv_layers'])
    max_layers = max(parent1['conv_layers'], parent2['conv_layers'])

    # Set both offspring to have the same number of layers initially
    offspring1['conv_layers'] = min_layers
    offspring2['conv_layers'] = min_layers

    # Decide layer counts for offspring
    # With 70% probability, use the layer count of one of the parents
    if random.random() < 0.7:
        if random.random() < 0.5:
            offspring1['conv_layers'] = parent1['conv_layers']
            offspring2['conv_layers'] = parent2['conv_layers']
        else:
            offspring1['conv_layers'] = parent2['conv_layers']
            offspring2['conv_layers'] = parent1['conv_layers']
    else:
        # With 30% probability, use a random value between min and max
        if min_layers != max_layers:
            offspring1['conv_layers'] = random.randint(min_layers, max_layers)
            offspring2['conv_layers'] = random.randint(min_layers, max_layers)

    # Possibly swap learning rates with 50% probability
    if random.random() < 0.5:
        offspring1['learning_rate'], offspring2['learning_rate'] = \
            offspring2['learning_rate'], offspring1['learning_rate']

    # For each possible layer up to the max of both offspring, handle parameter inheritance
    max_offspring_layers = max(offspring1['conv_layers'], offspring2['conv_layers'])

    for i in range(max_offspring_layers):
        # Determine which parents have this layer
        parent1_has_layer = i < parent1['conv_layers']
        parent2_has_layer = i < parent2['conv_layers']

        # For common layers (both parents have this layer), perform crossover
        if parent1_has_layer and parent2_has_layer:
            for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                param_name = f'{param}_{i}'

                # Randomly swap with 50% probability
                if random.random() < 0.5:
                    offspring1[param_name], offspring2[param_name] = \
                        offspring2[param_name], offspring1[param_name]

        # For layers only in parent1, inherit those parameters if needed
        elif parent1_has_layer:
            if i < offspring1['conv_layers']:  # offspring1 needs this layer
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    offspring1[f'{param}_{i}'] = parent1[f'{param}_{i}']

            if i < offspring2['conv_layers']:  # offspring2 needs this layer
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    offspring2[f'{param}_{i}'] = parent1[f'{param}_{i}']

        # For layers only in parent2, inherit those parameters if needed
        elif parent2_has_layer:
            if i < offspring1['conv_layers']:  # offspring1 needs this layer
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    offspring1[f'{param}_{i}'] = parent2[f'{param}_{i}']

            if i < offspring2['conv_layers']:  # offspring2 needs this layer
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    offspring2[f'{param}_{i}'] = parent2[f'{param}_{i}']

    return offspring1, offspring2


def mutate(
        individual: Dict[str, Any],
        mutation_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Mutate an individual by randomly changing some of its hyperparameters.

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
    # These are multipliers applied to the base mutation_rate
    scaling_factors = {
        'conv_layers': 0.5,     # Less likely to change architecture
        'learning_rate': 1.5,   # More likely to tune learning rate
        'filters': 1.2,         # Important but not drastic
        'kernel_size': 0.8,     # Less impact than filters
        'activation': 1.0,      # Normal importance
        'pool_type': 0.7,       # Less impact
        'dropout': 1.3          # Important for regularization
    }

    # Possibly mutate number of convolutional layers (with scaled probability)
    if random.random() < mutation_rate * scaling_factors['conv_layers']:
        # Get a new random number of layers from the search space
        new_conv_layers = random.choice(HYPERPARAMETER_SPACE['conv_layers'])

        # Handle the case where we're decreasing the number of layers
        if new_conv_layers < mutated['conv_layers']:
            # Remove parameters for the layers we're removing
            for i in range(new_conv_layers, mutated['conv_layers']):
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    key = f'{param}_{i}'
                    if key in mutated:
                        del mutated[key]

        # Handle the case where we're increasing the number of layers
        elif new_conv_layers > mutated['conv_layers']:
            # Add parameters for the new layers
            for i in range(mutated['conv_layers'], new_conv_layers):
                mutated[f'filters_{i}'] = random.choice(HYPERPARAMETER_SPACE['filters'])
                mutated[f'kernel_size_{i}'] = random.choice(HYPERPARAMETER_SPACE['kernel_sizes'])
                mutated[f'activation_{i}'] = random.choice(HYPERPARAMETER_SPACE['activation_functions'])
                mutated[f'pool_type_{i}'] = random.choice(HYPERPARAMETER_SPACE['pool_types'])
                mutated[f'dropout_{i}'] = random.choice(HYPERPARAMETER_SPACE['dropout_rates'])

        # Update the number of layers
        mutated['conv_layers'] = new_conv_layers

    # Possibly mutate learning rate (with scaled probability)
    if random.random() < mutation_rate * scaling_factors['learning_rate']:
        mutated['learning_rate'] = random.choice(HYPERPARAMETER_SPACE['learning_rates'])

    # Possibly mutate layer parameters
    for i in range(mutated['conv_layers']):
        # Apply scaled mutation rates for each parameter type

        # Mutate filters for this layer
        if random.random() < mutation_rate * scaling_factors['filters']:
            mutated[f'filters_{i}'] = random.choice(HYPERPARAMETER_SPACE['filters'])

        # Mutate kernel size for this layer
        if random.random() < mutation_rate * scaling_factors['kernel_size']:
            mutated[f'kernel_size_{i}'] = random.choice(HYPERPARAMETER_SPACE['kernel_sizes'])

        # Mutate activation function for this layer
        if random.random() < mutation_rate * scaling_factors['activation']:
            mutated[f'activation_{i}'] = random.choice(HYPERPARAMETER_SPACE['activation_functions'])

        # Mutate pool type for this layer
        if random.random() < mutation_rate * scaling_factors['pool_type']:
            mutated[f'pool_type_{i}'] = random.choice(HYPERPARAMETER_SPACE['pool_types'])

        # Mutate dropout rate for this layer
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

    This function combines selection, crossover, and mutation to create
    a new generation of individuals based on the fitness of the current population.

    Args:
        population (List[Dict[str, Any]]): Current generation of individuals
        fitness_scores (List[float]): Fitness scores for current population
        population_size (int): Desired size of the new population
        mutation_rate (float, optional): Probability of mutation.
                                         Default to 0.1.

    Returns:
        List[Dict[str, Any]]: New generation of individuals
    """
    # Create a new empty generation
    next_generation = []

    # Apply elitism: Keep the best individual(s)
    elite_count = max(1, int(population_size * 0.1))            # 10% elitism
    elite_indices = np.argsort(fitness_scores)[-elite_count:]   # Indices of top individuals

    for idx in elite_indices:
        next_generation.append(population[idx].copy())  # Use copy to avoid modifying the original

    # Helper function to check if an individual is too similar to existing ones
    def is_too_similar(individual, population, similarity_threshold=0.9):
        for existing in population:
            # Skip if different number of layers
            if existing['conv_layers'] != individual['conv_layers']:
                continue

            # Count identical parameters
            identical_params = 0
            total_params = 0

            # Check learning rate
            if existing['learning_rate'] == individual['learning_rate']:
                identical_params += 1
            total_params += 1

            # Check layer parameters
            for i in range(individual['conv_layers']):
                for param in ['filters', 'kernel_size', 'activation', 'pool_type', 'dropout']:
                    param_name = f'{param}_{i}'
                    if existing[param_name] == individual[param_name]:
                        identical_params += 1
                    total_params += 1

            # Calculate similarity ratio
            similarity = identical_params / total_params
            if similarity >= similarity_threshold:
                return True

        return False

    # Select parents for producing offspring
    parents = select_parents(population, fitness_scores, population_size)

    # Create offspring through crossover and mutation until we reach population_size
    max_attempts = population_size * 10  # Limit attempts to prevent infinite loops
    attempts = 0

    while len(next_generation) < population_size and attempts < max_attempts:
        attempts += 1

        # Select two parents
        parent1, parent2 = random.sample(parents, 2)

        # Create offspring through crossover
        offspring1, offspring2 = crossover(parent1, parent2)

        # Apply adaptive mutation based on similarity to parents
        # Use higher mutation rate if offspring is too similar to parents
        offspring1_mutation_rate = mutation_rate
        if is_too_similar(offspring1, [parent1, parent2], 0.8):
            offspring1_mutation_rate = min(mutation_rate * 2, 0.5)

        offspring2_mutation_rate = mutation_rate
        if is_too_similar(offspring2, [parent1, parent2], 0.8):
            offspring2_mutation_rate = min(mutation_rate * 2, 0.5)

        # Apply mutation
        offspring1 = mutate(offspring1, offspring1_mutation_rate)
        offspring2 = mutate(offspring2, offspring2_mutation_rate)

        # Only add offspring if they're not too similar to existing population
        if not is_too_similar(offspring1, next_generation) and len(next_generation) < population_size:
            next_generation.append(offspring1)

        if not is_too_similar(offspring2, next_generation) and len(next_generation) < population_size:
            next_generation.append(offspring2)

        # If we couldn't fill the population with diverse individuals,
        # fill remaining slots with random individuals
    while len(next_generation) < population_size:
        next_generation.append(generate_individual())

    return next_generation


def print_best_individual(individual: Dict[str, Any], fitness: float) -> None:
    """
    Print the hyperparameters and fitness of the best individual.

    This function provides a human-readable representation of CNN architecture
    and its performance.

    Args:
        individual (Dict[str, Any]): The individual to print
        fitness (float): The fitness (validation accuracy) of the individual
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

    This function visualizes the evolution progress by showing how the
    best fitness score changes over generations.

    Args:
        fitness_history (List[float]): List of best fitness scores from each generation
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.title('Fitness History')
    plt.grid(True)
    plt.show()


def run_genetic_algorithm(
        population_size: int = 10,
        num_generations: int = 10,
        mutation_rate: float = 0.1,
        epochs_per_eval: int = 5
) -> Tuple[Dict[str, Any], float, List[float]]:
    """
    Run the genetic algorithm to optimize CNN hyperparameters.

    This is the main function that orchestrates the entire genetic algorithm process.

    Process:
    1. Load and preprocess the data
    2. Initialize a random population
    3. For each generation:
       a. Evaluate the fitness of each individual
       b. Track the best individual
       c. Evolve the population to create the next generation
    4. Return the best individual found, its fitness, and the fitness history

    Args:
        population_size (int, optional): Number of individuals in each generation.
                                         Defaults to 10.
        num_generations (int, optional): Number of generations to evolve.
                                         Defaults to 10.
        mutation_rate (float, optional): Probability of mutation for each hyperparameter.
                                         Defaults to 0.1.
        epochs_per_eval (int, optional): Number of training epochs for each fitness evaluation.
                                         Defaults to 5.

    Returns:
        Tuple[Dict[str, Any], float, List[float]]:
            - Best individual (hyperparameters)
            - Best fitness (validation accuracy)
            - Fitness history (list of best fitness values in each generation)
    """
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

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
            fitness = evaluate_fitness(individual, x_train, y_train, x_val, y_val, epochs=epochs_per_eval)
            fitness_scores.append(fitness)

        # Find the best individual in this generation
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_individual = population[gen_best_idx]

        # Update overall best if better
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual

        # Add to fitness history
        fitness_history.append(gen_best_fitness)

        # Print generation stats
        elapsed_time = time.time() - start_time
        print(f"  Best fitness in generation: {gen_best_fitness:.4f}")
        print(f"  Overall best fitness: {best_fitness:.4f}")
        print(f"  Time taken: {elapsed_time:.2f} seconds")

        # Stop if we've reached the last generation
        if generation == num_generations - 1:
            break

        # Evolve population
        population = evolve_population(population, fitness_scores, population_size, mutation_rate)

    return best_individual, best_fitness, fitness_history


# Main execution
if __name__ == "__main__":
    # Run genetic algorithm
    best_individual, best_fitness, fitness_history = run_genetic_algorithm(
        population_size=10,
        num_generations=10,
        mutation_rate=0.1,
        epochs_per_eval=3  # Use a small number for faster evaluation
    )

    # Print results
    print("\nGenetic Algorithm Results:")
    print_best_individual(best_individual, best_fitness)

    # Plot fitness history
    plot_fitness_history(fitness_history)

    # Optional: Evaluate the best model on the test set with more epochs
    print("\nEvaluating best model on test set...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    best_model = build_model(best_individual)

    # Combine training and validation data for final training
    x_train_full = np.concatenate([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Train with more epochs
    best_model.fit(
        x_train_full, y_train_full,
        epochs=10,
        batch_size=64,
        verbose=1
    )

    # Evaluate on the test set
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=1)
    print(f"Final test accuracy: {test_accuracy:.4f}")

    # TODO: Add code for comparison with baseline methods (grid search or random search)
    # This would involve implementing grid search and random search methods
    # and comparing their performance with the genetic algorithm in terms of:
    # - Best architecture found
    # - Time taken
    # - Number of architectures evaluated

    # TODO: Add code for visualizing the best CNN architecture
    # This could involve creating a diagram of the CNN layers
    # or using TensorFlow's model visualization capabilities

    # TODO: Add code for ablation studies
    # This would involve running experiments with different genetic algorithm parameters
    # such as different mutation rates, selection strategies, or crossover techniques
    # to understand their impact on the optimization process
