# Hyperparameter Tuning for CNNs Using Genetic Algorithms

## Overview / Summary

This report documents the development and evaluation of a genetic algorithm framework for optimizing hyperparameters of Convolutional Neural Networks (CNNs) on the CIFAR-10 image classification dataset. Traditional hyperparameter tuning methods like grid search and random search can be computationally expensive and scale poorly with the number of hyperparameters. Genetic algorithms provide a biologically-inspired approach that can more effectively navigate complex hyperparameter spaces.

The implementation demonstrates that genetic algorithms can efficiently discover high-performing CNN architectures, consistently outperforming both random search and grid search given the same computational budget. Through systematic ablation studies, we also identified which components and parameters of the genetic algorithm have the most significant impact on performance.

The framework developed in this project enables automatic design of CNN architectures, reducing the need for manual hyperparameter tuning and potentially leading to better and more efficient neural network models for image classification tasks.

## 1. Introduction

### 1.1 Problem Statement

Convolutional Neural Networks (CNNs) have become the standard approach for image classification tasks. However, designing effective CNN architectures remains challenging due to the vast search space of hyperparameters. Manual tuning is time-consuming and may result in suboptimal architectures, while traditional methods like grid search and random search scale poorly with the number of hyperparameters.

The key challenge addressed in this project is: **How can we automatically discover optimal or near-optimal CNN architectures for a given image classification task?**

### 1.2 Objectives

The main objectives of this project were to:

1. Implement a genetic algorithm framework for CNN hyperparameter tuning
2. Compare the genetic algorithm approach with traditional methods (grid search, random search)
3. Analyze the impact of different genetic algorithm components and parameters
4. Understand the evolutionary trajectory of CNN architectures
5. Visualize and interpret the results

### 1.3 CIFAR-10 Dataset

CIFAR-10 consists of 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. For our experiments, I further divided the training set into 45,000 training images and 5,000 validation images, using the validation set for fitness evaluation during the genetic algorithm.

![CIFAR-10 Sample Images](figures/cifar10_samples.png)
*Figure 1: Sample images from the CIFAR-10 dataset showing examples from each class.*

## 2. Methodology

### 2.1 Genetic Algorithm Overview

Genetic algorithms are optimization techniques inspired by the process of natural selection. The key components include:

1. **Representation**: Each individual represents a set of CNN hyperparameters.
2. **Initialization**: Create an initial population of random CNN architectures.
3. **Fitness Evaluation**: Train and evaluate each CNN architecture on the validation set.
4. **Selection**: Choose the fittest individuals to become parents for the next generation.
5. **Crossover**: Combine hyperparameters from two parents to create offspring.
6. **Mutation**: Randomly change some hyperparameters to maintain diversity.
7. **Replacement**: Form a new generation from the offspring and possibly some parents.

![Genetic Algorithm Flow](figures/genetic_algorithm_flow.png)
*Figure 2: Flow diagram of the genetic algorithm for CNN hyperparameter tuning.*

### 2.2 Hyperparameter Search Space

I defined a comprehensive hyperparameter search space for our CNN architectures:

| Hyperparameter | Values |
|----------------|--------|
| Number of convolutional layers | 1, 2, 3, 4, 5 |
| Number of filters per layer | 16, 32, 64, 128, 256 |
| Kernel sizes | 3, 5, 7 |
| Pooling types | max, avg, none |
| Learning rates | 0.1, 0.01, 0.001, 0.0001 |
| Activation functions | relu, elu, leaky_relu |
| Dropout rates | 0.0, 0.25, 0.5 |

### 2.3 CNN Model Architecture

The base CNN architecture consisted of:
- Variable number of convolutional layers (determined by the genetic algorithm)
- Each convolutional layer followed by optional pooling and dropout (determined by the genetic algorithm)
- A flattening layer
- A fully connected layer with 128 units and ReLU activation
- A dropout layer with rate 0.5
- An output layer with 10 units (one for each CIFAR-10 class) and softmax activation

![CNN Architecture](figures/cnn_architecture.png)
*Figure 3: Example CNN architecture with hyperparameters determined by the genetic algorithm.*

### 2.4 Genetic Algorithm Implementation

#### 2.4.1 Individual Representation

Each individual (CNN architecture) was represented as a dictionary containing:
- The number of convolutional layers
- The learning rate
- For each convolutional layer:
  - Number of filters
  - Kernel size
  - Activation function
  - Pooling type
  - Dropout rate

```python
# Example individual with 2 convolutional layers
{
    'conv_layers': 2,
    'learning_rate': 0.001,
    'filters_0': 64,
    'kernel_size_0': 3,
    'activation_0': 'relu',
    'pool_type_0': 'max',
    'dropout_0': 0.25,
    'filters_1': 128,
    'kernel_size_1': 5,
    'activation_1': 'elu',
    'pool_type_1': 'avg',
    'dropout_1': 0.0
}
```

#### 2.4.2 Population Initialization

Initialize the population by generating random individuals, each with randomly selected hyperparameters from the search space.

#### 2.4.3 Fitness Evaluation

The fitness of each individual was evaluated by:
1. Building a CNN model with the individual's hyperparameters
2. Training the model on the training set for a fixed number of epochs
3. Evaluating the trained model on the validation set
4. Using the validation accuracy as the fitness score

#### 2.4.4 Selection Strategy

Touranment selection was the strategy implemented: 
1. Randomly select k individuals from the population
2. Choose the individual with the highest fitness as a parent
3. Repeat until we have the desired number of parents

![Tournament Selection](figures/tournament_selection.png)
*Figure 4: Illustration of tournament selection with tournament size k=3.*

#### 2.4.5 Crossover Strategy

For crossover, a variant of uniform crossover is implemented:
1. For each hyperparameter, randomly decide which parent to inherit from
2. Handle special cases for the number of convolutional layers
3. Create two offspring from each pair of parents

![Crossover Strategy](figures/crossover_strategy.png)
*Figure 5: Uniform crossover applied to CNN hyperparameters.*

#### 2.4.6 Mutation Strategy

The mutation strategy used parameter-specific mutation rates:
1. Assign different mutation probabilities to different hyperparameters
2. For each hyperparameter, decide whether to mutate based on its mutation probability
3. If mutating, replace the value with a random value from the search space
4. Handle special cases when changing the number of layers

![Mutation Strategy](figures/mutation_strategy.png)
*Figure 6: Mutation with parameter-specific mutation rates.*

#### 2.4.7 Replacement Strategy

Combined elitiscm with generational replacement:
1. Keep the best 1-2 individuals from the current generation (depending on population size)
2. Fill the rest of the new generation with offspring created through selection, crossover, and mutation

### 2.5 Baseline Methods

I implemented two baseline methods for comparison:

#### 2.5.1 Random Search

1. Generate random hyperparameter combinations from the search space
2. Evaluate each combination on the validation set
3. Return the best combination found

#### 2.5.2 Grid Search

1. Define a reduced hyperparameter space to make grid search computationally feasible
2. Create a grid of all combinations of hyperparameters
3. Evaluate each combination on the validation set
4. Return the best combination found

### 2.6 Evaluation Metrics

The following metrics were used to evaluate the approach:

1. **Validation Accuracy**: The primary fitness metric during evolution
2. **Test Accuracy**: Final evaluation metric for the best architectures
3. **Computational Efficiency**: Time taken per evaluation
4. **Convergence Speed**: How quickly the fitness improves over generations

## 3. Implementation and Experiments

### 3.1 Implementation Details

The implementation was done in Python using TensorFlow for the CNN models and NumPy for the genetic algorithm components. The codebase consists of:

1. **Core Genetic Algorithm Components**:
   - `generate_individual()`: Creates random hyperparameter sets
   - `initialize_population()`: Initializes a population of random individuals
   - `evaluate_fitness()`: Evaluates CNN architectures on the validation set
   - `select_parents()`: Selects parents using tournament selection
   - `crossover()`: Combines hyperparameters from two parents
   - `mutate()`: Randomly changes hyperparameters
   - `evolve_population()`: Creates a new generation from the current one

2. **Model Building and Evaluation**:
   - `build_model()`: Constructs a CNN based on hyperparameters
   - `run_genetic_algorithm()`: Orchestrates the entire evolutionary process

3. **Baseline Methods**:
   - `run_random_search()`: Implements random hyperparameter search
   - `run_grid_search()`: Implements grid search for hyperparameters

4. **Visualization and Analysis**:
   - `visualize_filters()`: Visualizes CNN filters
   - `visualize_feature_maps()`: Visualizes feature maps for a given image
   - `analyze_hyperparameter_impact()`: Analyzes how hyperparameters affect performance
   - `analyze_population_evolution()`: Tracks diversity and fitness over generations

### 3.2 Experimental Setup

We conducted experiments with the following parameters:

- **Genetic Algorithm**:
  - Population size: 8-10 individuals
  - Generations: 5-10
  - Tournament size: 3
  - Mutation rate: 0.1-0.2
  - Epochs per fitness evaluation: 3

- **Random Search**:
  - Number of evaluations: Same as genetic algorithm (population_size × generations)
  - Epochs per evaluation: 3

- **Grid Search**:
  - Number of evaluations: Same as genetic algorithm
  - Reduced hyperparameter space to make it computationally feasible
  - Epochs per evaluation: 3

### 3.3 Ablation Studies

Ablation studies were coducted to understand the impact of different components and parameters:

1. **Selection Strategies**: Compared tournament selection, roulette wheel selection, and rank selection
2. **Crossover Strategies**: Compared single-point crossover, uniform crossover, and arithmetic recombination
3. **Mutation Strategies**: Compared standard mutation, Gaussian mutation, and adaptive mutation
4. **Population Size**: Tested sizes from 5 to 20
5. **Number of Generations**: Tested from 5 to 15 generations
6. **Mutation Rate**: Tested rates from 0.05 to 0.3
7. **Tournament Size**: Tested sizes from 2 to 5
8. **Elitism Count**: Tested from 1 to 3 elite individuals

## 4. Results and Analysis

### 4.1 Genetic Algorithm Performance

The genetic algorithm showed consistent improvement in fitness over generations. The best architecture achieved a validation accuracy of 0.72 and a test accuracy of 0.69 after just 5 generations with a population size of 8. The final CNN architecture had 3 convolutional layers with the following characteristics:

- **Layer 1**: 64 filters, 3×3 kernel, ReLU activation, max pooling, 0.25 dropout
- **Layer 2**: 128 filters, 3×3 kernel, ReLU activation, max pooling, 0.25 dropout
- **Layer 3**: 64 filters, 5×5 kernel, ELU activation, average pooling, 0.5 dropout
- **Learning rate**: 0.001

![Fitness History](figures/fitness_history.png)
*Figure 7: Fitness improvement over generations showing the effectiveness of the genetic algorithm.*

The fitness history showed a clear upward trend, indicating that the genetic algorithm was effectively exploring the hyperparameter space and evolving better architectures over time.

### 4.2 Comparison with Baseline Methods

| Method | Best Validation Accuracy | Evaluations | Time per Evaluation (s) | Total Time (s) |
|--------|--------------------------|------------|-------------------------|----------------|
| Genetic Algorithm | 0.72 | 40 | 18.5 | 740 |
| Random Search | 0.67 | 40 | 17.8 | 712 |
| Grid Search | 0.65 | 40 | 17.9 | 716 |

![Method Comparison](figures/method_comparison.png)
*Figure 8: Comparison of validation accuracy and computational efficiency across methods.*

The genetic algorithm outperformed both random search and grid search despite evaluating the same number of architectures. This suggests that the evolutionary approach was more effective at navigating the hyperparameter space than random sampling or systematic grid search.

### 4.3 Hyperparameter Impact Analysis

Based on the analysis of the hyperparameter impact on validation accuracy:

![Hyperparameter Impact](figures/hyperparameter_impact.png)
*Figure 9: Impact of different hyperparameters on validation accuracy.*

1. **Number of Convolutional Layers**: Models with 3-4 layers performed best. Too few layers couldn't capture complex patterns, while too many layers led to overfitting or vanishing gradients.

2. **Learning Rate**: A learning rate of 0.001 generally worked best, balancing fast convergence with stability. Higher learning rates (0.1, 0.01) often led to unstable training, while lower rates (0.0001) converged too slowly.

3. **Filters**: Higher filter counts generally improved performance, with 64-128 filters showing the best results. However, very high filter counts (256) often led to overfitting and longer training times.

4. **Kernel Size**: Smaller kernels (3×3) generally performed better than larger ones (5×5, 7×7), likely because they capture local features more effectively while using fewer parameters.

5. **Pooling Type**: MaxPooling slightly outperformed AveragePooling, while "none" (no pooling) performed worse. This suggests that downsampling is important for CIFAR-10, with MaxPooling's ability to capture the most prominent features being particularly beneficial.

6. **Activation Function**: ReLU performed best overall, followed by ELU and LeakyReLU. This aligns with common practices in CNN design.

7. **Dropout Rate**: Moderate dropout (0.25) performed better than no dropout (0.0) or high dropout (0.5). This indicates that some regularization is helpful, but excessive dropout can hinder learning.

![Correlation Matrix](figures/correlation_matrix.png)
*Figure 10: Correlation matrix showing relationships between hyperparameters and validation accuracy.*

### 4.4 Ablation Study Results

The ablation studies revealed:

![Ablation Studies](figures/ablation_studies.png)
*Figure 11: Results of ablation studies for different genetic algorithm components.*

1. **Selection Strategy**: Tournament selection performed best, followed by rank selection and then roulette wheel selection. Tournament selection's balance of selection pressure and diversity maintenance likely explains its effectiveness.

2. **Crossover Strategy**: Uniform crossover outperformed single-point crossover and arithmetic recombination. The uniform approach allows for more flexible recombination of hyperparameters.

3. **Mutation Strategy**: Adaptive mutation performed best, followed by Gaussian mutation and standard mutation. The adaptive approach's ability to balance exploration in early generations with exploitation in later generations proved effective.

4. **Population Size**: Larger populations (15-20) generally found better solutions but required more computational resources. Population sizes of 8-10 offered a good balance between exploration and computational efficiency.

5. **Number of Generations**: More generations consistently led to better results, with significant improvements in the first 5 generations and more gradual improvements thereafter.

6. **Mutation Rate**: Optimal rates were in the 0.1-0.2 range. Lower rates led to premature convergence, while higher rates disrupted good solutions too frequently.

7. **Tournament Size**: A tournament size of 3 worked best. Smaller sizes reduced selection pressure, while larger sizes led to loss of diversity.

8. **Elitism Count**: Keeping 1-2 elite individuals proved optimal for our population sizes. More elitism reduced exploration, while no elitism risked losing the best solutions.

### 4.5 Population Diversity Analysis

Analysis of how population diversity evolved over generations to understand the balance between exploration and exploitation in our genetic algorithm.

![Population Diversity](figures/population_diversity.png)
*Figure 12: Population diversity metrics over generations showing convergence behavior.*

The analysis revealed:
1. **Hyperparameter Diversity**: The diversity of hyperparameters decreased over generations as the population converged toward optimal values.
2. **Fitness Distribution**: The fitness distribution became narrower and shifted toward higher values in later generations.
3. **Architectural Patterns**: Certain architectural patterns (e.g., 3-4 layers with decreasing filter sizes) emerged consistently in later generations.

This analysis confirms that our genetic algorithm was effectively balancing exploration and exploitation, starting with diverse architectures and gradually converging toward optimal solutions.

### 4.6 CNN Feature Visualization

Visualizing the filters and feature maps of our best CNN architecture revealed:

![CNN Filters](figures/cnn_filters.png)
*Figure 13: Visualization of filters from the first convolutional layer of the best CNN architecture.*

1. **Filters**: The first layer learned edge detectors and color filters, as expected. Deeper layers developed more complex pattern detectors.

![Feature Maps](figures/feature_maps.png)
*Figure 14: Feature maps showing activations at different layers for a sample image.*

2. **Feature Maps**: Lower layers activated on simple patterns and edges, while deeper layers showed activations for more complex, class-specific features.

This visualization confirmed that our CNN was learning meaningful hierarchical representations of the CIFAR-10 images.

### 4.7 Classification Performance

To evaluate the generalization capability of our best architecture, we analyzed its performance on the test set using a confusion matrix and class-specific metrics.

![Confusion Matrix](figures/confusion_matrix.png)
*Figure 15: Confusion matrix showing the classification performance across CIFAR-10 classes.*

The confusion matrix revealed:
1. **Strong Classes**: The model performed particularly well on automobile, ship, and truck classes.
2. **Challenging Classes**: The model had more difficulty distinguishing between cat, dog, and deer classes.
3. **Common Confusions**: The most frequent confusions were between semantically similar classes (e.g., cat vs. dog, automobile vs. truck).

Overall, the model achieved a balanced accuracy across classes, indicating that it had learned meaningful features for all categories rather than biasing toward specific ones.

## 5. Conclusions and Future Work

### 5.1 Conclusions

1. **Effectiveness of Genetic Algorithms**: Genetic algorithms are an effective approach for CNN hyperparameter tuning, consistently outperforming random search and grid search with the same evaluation budget.

2. **Evolutionary Dynamics**: The genetic algorithm displayed clear evolutionary dynamics, with increasing population fitness over generations and convergence toward optimal architectures.

3. **Parameter Sensitivity**: The genetic algorithm's performance was most sensitive to the selection strategy, mutation rate, and population size. Tournament selection, moderate mutation rates (0.1-0.2), and adequate population sizes (8-10) offered the best results.

4. **Optimal CNN Architecture**: For CIFAR-10, the genetic algorithm discovered that architectures with 3-4 convolutional layers, moderate filter counts (64-128), small kernel sizes (3×3), max pooling, and moderate dropout (0.25) performed best.

5. **Computational Efficiency**: While the genetic algorithm had a slightly higher computational cost per evaluation due to population management, it reached better solutions faster by leveraging evolutionary principles.

### 5.2 Future Work

1. **Extended Parameter Space**: Future work could explore a larger hyperparameter space, including different layer types (e.g., depthwise convolutions, residual connections) and architecture patterns.

2. **Advanced Genetic Operators**: Implementing more sophisticated selection, crossover, and mutation strategies could potentially improve performance further.

3. **Multi-objective Optimization**: Extending the approach to optimize for both performance and computational efficiency simultaneously would be valuable for resource-constrained applications.

4. **Transfer Learning**: Incorporating transfer learning from pre-trained models into the genetic algorithm framework could significantly improve results and reduce training time.

5. **Neural Architecture Search**: Expanding beyond hyperparameter tuning to full neural architecture search would allow even more flexible exploration of the design space.

6. **Longer Evolution**: Running the genetic algorithm for more generations and with larger populations would likely yield even better results, though at increased computational cost.

## Appendix A: Detailed Implementation

```python
def run_genetic_algorithm(
    population_size: int = 10,
    num_generations: int = 10,
    mutation_rate: float = 0.1,
    epochs_per_eval: int = 5
):
    """Run the genetic algorithm to optimize CNN hyperparameters."""
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
        print(f"\nGeneration {generation+1}/{num_generations}")

        # Evaluate fitness for each individual
        fitness_scores = []
        for i, individual in enumerate(population):
            fitness = evaluate_fitness(
                individual, x_train, y_train, x_val, y_val, 
                epochs=epochs_per_eval
            )
            fitness_scores.append(fitness)

        # Find best individual in this generation
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
        print(f"  Best fitness in generation: {gen_best_fitness:.4f}")
        print(f"  Overall best fitness: {best_fitness:.4f}")

        # If we've reached the last generation, we're done
        if generation == num_generations - 1:
            break

        # Evolve population
        population = evolve_population(
            population, fitness_scores, population_size, mutation_rate
        )

    return best_individual, best_fitness, fitness_history
```

## Appendix B: Detailed Ablation Study Results

| Component/Parameter | Variation | Best Validation Accuracy | Notes |
|---------------------|-----------|--------------------------|-------|
| Selection Strategy | Tournament | 0.72 | Best overall performance |
| Selection Strategy | Roulette Wheel | 0.68 | Less stable, sensitive to fitness scaling |
| Selection Strategy | Rank | 0.70 | More stable than roulette, but slower convergence |
| Crossover Strategy | Single-Point | 0.69 | Simpler implementation but less flexible |
| Crossover Strategy | Uniform | 0.72 | Best overall performance, more flexible recombination |
| Crossover Strategy | Arithmetic | 0.67 | Good for numerical parameters, worse for categorical ones |
| Mutation Strategy | Standard | 0.68 | Simple but less adaptive |
| Mutation Strategy | Gaussian | 0.70 | Better for numerical parameters |
| Mutation Strategy | Adaptive | 0.72 | Best overall, balances exploration and exploitation |
| Population Size | 5 | 0.65 | Insufficient diversity |
| Population Size | 10 | 0.72 | Good balance of diversity and efficiency |
| Population Size | 20 | 0.73 | Slightly better results but much higher computational cost |
| Generations | 5 | 0.70 | Good initial progress |
| Generations | 10 | 0.72 | Continued improvement |
| Generations | 15 | 0.74 | Diminishing returns after 10 generations |
| Mutation Rate | 0.05 | 0.67 | Insufficient exploration |
| Mutation Rate | 0.1 | 0.72 | Good balance |
| Mutation Rate | 0.2 | 0.71 | Slightly too disruptive |
| Mutation Rate | 0.3 | 0.68 | Too disruptive |
| Tournament Size | 2 | 0.69 | Low selection pressure |
| Tournament Size | 3 | 0.72 | Optimal balance |
| Tournament Size | 5 | 0.70 | Too high selection pressure, loss of diversity |
| Elitism Count | 1 | 0.71 | Minimal elitism |
| Elitism Count | 2 | 0.72 | Optimal for our population size |
| Elitism Count | 3 | 0.70 | Too much elitism, reduced exploration |