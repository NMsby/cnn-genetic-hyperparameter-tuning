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
    """Load and preprocess CIFAR-10 dataset."""
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

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# Define hyperparameter search space
HYPERPARAMETER_SPACE = {
    'conv_layers': [1, 2, 3, 4, 5],                          # Number of convolutional layers
    'filters': [16, 32, 64, 128, 256],                       # Number of filters per layer
    'kernel_sizes': [3, 5, 7],                               # Kernel sizes for convolutional layers
    'pool_types': ['max', 'avg', 'none'],                    # Pooling types
    'learning_rates': [0.1, 0.01, 0.001, 0.0001],            # Learning rates
    'activation_functions': ['relu', 'elu', 'leaky_relu'],   # Activation functions
    'dropout_rates': [0.0, 0.25, 0.5]                        # Dropout rates
}

# Model builder function
def build_model(hyperparameters: Dict[str, Any]) -> tf.keras.Model:
    """Build a CNN model based on provided hyperparameters."""
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
                input_shape=(32, 32, 3)
            ))
        else:
            model.add(Conv2D(
                filters=hyperparameters[f'filters_{i}'],
                kernel_size=hyperparameters[f'kernel_size_{i}'],
                activation=hyperparameters[f'activation_{i}'],
                padding='same'
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

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Function to flatten hyperparameters from the search space into model parameters
def generate_individual() -> Dict[str, Any]:
    """Generate a random individual (set of hyperparameters)."""
    # TODO: Student implementation
    # Create a dictionary with all necessary hyperparameters
    # Example structure:
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
    """Initialize a population of random individuals."""
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

    This serves as the fitness function for our genetic algorithm.
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
        verbose=0
    )

    # Return validation accuracy as fitness
    return history.history['val_accuracy'][-1]

def select_parents(
    population: List[Dict[str, Any]],
    fitness_scores: List[float],
    num_parents: int
) -> List[Dict[str, Any]]:
    """
    Select individuals to be parents for the next generation using tournament selection.

    Tournament selection: randomly select k individuals and choose the best one.
    Repeat until we have num_parents parents.
    """
    # TODO: Student implementation
    parents = []
    # Example of tournament selection:
    tournament_size = 3  # You can adjust this parameter

    for _ in range(num_parents):
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(range(len(population)), tournament_size)
        # Get their fitness scores
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        # Find the winner (highest fitness)
        winner_relative_idx = np.argmax(tournament_fitness)
        winner_idx = tournament_indices[winner_relative_idx]
        # Add the winner to parents
        parents.append(population[winner_idx])

    return parents

def crossover(
    parent1: Dict[str, Any],
    parent2: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform crossover between two parents to create two offspring.

    This implementation should mix hyperparameters from both parents.
    """
    # TODO: Student implementation
    # Create copies of parents to avoid modifying them
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Implement crossover logic here
    # Example: simple one-point crossover for some parameters

    return offspring1, offspring2

def mutate(
    individual: Dict[str, Any],
    mutation_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Mutate an individual by randomly changing some of its hyperparameters.

    The mutation_rate determines the probability of each hyperparameter being mutated.
    """
    # TODO: Student implementation
    # Create a copy to avoid modifying the original
    mutated = individual.copy()

    # Implement mutation logic here
    # For each hyperparameter, with probability mutation_rate, change it to a random value

    return mutated

def evolve_population(
    population: List[Dict[str, Any]],
    fitness_scores: List[float],
    population_size: int,
    mutation_rate: float = 0.1
) -> List[Dict[str, Any]]:
    """Evolve the population to the next generation."""
    # TODO: Student implementation
    # Select parents
    num_parents = population_size // 2
    parents = select_parents(population, fitness_scores, num_parents)

    # Create offspring through crossover and mutation
    next_generation = []

    # Implement the logic to create the next generation using crossover and mutation

    return next_generation

def print_best_individual(individual: Dict[str, Any], fitness: float) -> None:
    """Print the hyperparameters and fitness of the best individual."""
    print(f"Best fitness (validation accuracy): {fitness:.4f}")
    print("Best hyperparameters:")
    print(f"  - Number of conv layers: {individual['conv_layers']}")
    print(f"  - Learning rate: {individual['learning_rate']}")

    for i in range(individual['conv_layers']):
        print(f"  - Layer {i+1}:")
        print(f"    - Filters: {individual[f'filters_{i}']}")
        print(f"    - Kernel size: {individual[f'kernel_size_{i}']}")
        print(f"    - Activation: {individual[f'activation_{i}']}")
        print(f"    - Pool type: {individual[f'pool_type_{i}']}")
        print(f"    - Dropout: {individual[f'dropout_{i}']}")

def plot_fitness_history(fitness_history: List[float]) -> None:
    """Plot the fitness history of the best individual in each generation."""
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

    Returns:
        Tuple containing:
        - Best individual (hyperparameters)
        - Best fitness (validation accuracy)
        - Fitness history (list of best fitness values in each generation)
    """
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Initialize population
    population = initialize_population(population_size)

    # Track best individual and fitness history
    best_individual = None
    best_fitness = 0
    fitness_history = []

    # Main loop
    for generation in range(num_generations):
        start_time = time.time()
        print(f"\nGeneration {generation+1}/{num_generations}")

        # Evaluate fitness for each individual
        fitness_scores = []
        for i, individual in enumerate(population):
            print(f"  Evaluating individual {i+1}/{population_size}...", end="\r")
            fitness = evaluate_fitness(individual, x_train, y_train, x_val, y_val, epochs=epochs_per_eval)
            fitness_scores.append(fitness)

        # Find best individual in this generation
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

    # Evaluate on test set
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=1)
    print(f"Final test accuracy: {test_accuracy:.4f}")

    # TODO: Add code for comparison with baseline methods (grid search or random search)

    # TODO: Add code for visualizing the best CNN architecture

    # TODO: Add code for ablation studies