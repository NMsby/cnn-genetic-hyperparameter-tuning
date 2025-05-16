# CNN Hyperparameter Tuning with Genetic Algorithms

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A framework for optimizing Convolutional Neural Network (CNN) hyperparameters using genetic algorithms, applied to the CIFAR-10 image classification dataset. This project demonstrates how evolutionary approaches can efficiently navigate complex hyperparameter spaces and discover high-performing CNN architectures.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameter Space](#hyperparameter-space)
- [Genetic Algorithm Components](#genetic-algorithm-components)
- [Experimental Results](#experimental-results)
- [Performance Optimization](#performance-optimization)
- [Ablation Studies](#ablation-studies)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

Designing effective CNN architectures remains challenging due to the vast search space of hyperparameters. Traditional methods like grid search and random search scale poorly with the number of hyperparameters. This project implements a genetic algorithm approach that:

1. Automatically discovers near-optimal CNN architectures for image classification
2. Efficiently explores the hyperparameter space using evolutionary principles
3. Delivers superior results compared to traditional methods given the same computational budget
4. Provides insights into the impact of different hyperparameters on model performance

The genetic algorithm evolves a population of CNN architectures over multiple generations, using validation accuracy as the fitness metric. Through selection, crossover, and mutation, the algorithm converges toward increasingly better architectures.

## ✨ Features

- **Complete Genetic Algorithm Framework**: Implementation of all core genetic algorithm components for hyperparameter tuning
- **Performance Optimization**: Parallel processing, caching, and early stopping for efficient execution
- **Baseline Comparisons**: Implementation of random search and grid search for comparison
- **Ablation Studies**: Systematic analysis of different genetic algorithm components and parameters
- **Comprehensive Visualization**: Tools for visualizing CNN architectures, filters, feature maps, and evolutionary progress
- **Modular Design**: Clean separation of core components, experiments, and visualization tools
- **Command-line Interface**: Unified CLI for running different components of the project
- **Extensive Documentation**: Detailed documentation, comments, and type hints throughout the codebase

## 📁 Project Structure

```
cnn-genetic-hyperparameter-tuning/
├── data/                   # Data storage
├── docs/                   # Documentation
│   ├── figures/            # Figures for documentation
│   └── report.md           # Final report
├── logs/                   # Log files
│   ├── ablation_studies.log
│   ├── baseline_comparison.log
│   └── test_run.log
├── notebooks/              # Jupyter notebooks
│   └── cnn_genetic_tuning_analysis.ipynb  # Analysis notebook
├── results/                # Results storage
│   ├── ablation/           # Ablation study results
│   ├── baseline/           # Baseline comparison results
│   ├── figures/            # Generated figures
│   └── models/             # Saved models
├── src/                    # Source code
│   ├── core/               # Core implementation
│   │   ├── __init__.py
│   │   ├── genetic_algorithm.py  # Main GA implementation
│   │   ├── model_builder.py      # CNN model builder
│   │   └── utils.py              # Utility functions
│   ├── experiments/        # Experiment scripts
│   │   ├── __init__.py
│   │   ├── ablation_studies.py   # Ablation studies
│   │   ├── baseline_methods.py   # Baseline comparison methods
│   │   └── optimized_algorithm.py  # Optimized implementation
│   ├── utils/              # Utility scripts
│   │   ├── __init__.py
│   │   └── verify_tensorflow.py  # TensorFlow verification
│   ├── visualization/      # Visualization code
│   │   ├── __init__.py
│   │   ├── architecture_viz.py   # Architecture visualization
│   │   └── results_viz.py        # Results visualization
│   ├── __init__.py
│   └── genetic_algorithms_starter.py     # Starter code        
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── run_tests.py        # Test runner
│   ├── test_crossover.py
│   ├── test_evolution.py
│   └── ... (other test files)
├── .gitignore              # Git ignore file
├── environment.yml         # Conda environment
├── LICENSE                 # License file
├── main.py                 # Entry point script
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── setup.bat               # Windows setup script
├── setup.py                # Installation script
└── setup.sh                # Unix setup script
```

## 💻 Installation

### Prerequisites

- Python 3.10 or higher
- TensorFlow 2.19.0 or higher
- CUDA-compatible GPU (recommended but not required)

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/cnn-genetic-hyperparameter-tuning.git
cd cnn-genetic-hyperparameter-tuning

# Create and activate the conda environment
conda env create -f environment.yml
conda activate cnn-genetic

# Install the package in development mode
pip install -e .
```

### Option 2: Using Pip

```bash
# Clone the repository
git clone https://github.com/yourusername/cnn-genetic-hyperparameter-tuning.git
cd cnn-genetic-hyperparameter-tuning

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 3: Using Setup Scripts

For Unix/Linux/Mac:
```bash
git clone https://github.com/yourusername/cnn-genetic-hyperparameter-tuning.git
cd cnn-genetic-hyperparameter-tuning
chmod +x setup.sh
./setup.sh
```

For Windows:
```bash
git clone https://github.com/yourusername/cnn-genetic-hyperparameter-tuning.git
cd cnn-genetic-hyperparameter-tuning
setup.bat
```

## 🚀 Usage

### Command-line Interface

The project provides a unified command-line interface through `main.py`:

#### Run the Genetic Algorithm

```bash
# Basic usage
python main.py ga --population 10 --generations 10

# With custom parameters
python main.py ga --population 20 --generations 15 --mutation_rate 0.2 --epochs 5 --batch_size 64 --evaluate
```

#### Run the Optimized (Parallel) Genetic Algorithm

```bash
# With 4 parallel workers
python main.py optimized --population 10 --generations 10 --workers 4

# Full example
python main.py optimized --population 15 --generations 10 --mutation_rate 0.15 --epochs 3 --batch_size 128 --workers 4 --evaluate
```

#### Run Baseline Methods

```bash
# Run random search
python main.py baseline --method random --samples 20 --epochs 3

# Run grid search
python main.py baseline --method grid --samples 20 --epochs 3

# Run comparison of all methods
python main.py baseline --method all --samples 40 --epochs 3
```

#### Run Ablation Studies

```bash
# Run multiple ablation studies
python main.py ablation --parallel

# Run focused ablation study
python main.py ablation --focused

# Visualize existing ablation results
python main.py ablation --visualize
```

#### Run Visualizations

```bash
# Visualize CNN architecture
python main.py visualize --type architecture

# Visualize features of a trained model
python main.py visualize --type features --model_path results/ga_results.json

# Visualize experimental results
python main.py visualize --type results

# Run all visualizations
python main.py visualize --type all
```

#### Run Benchmarks

```bash
# Compare optimized vs original implementation
python main.py benchmark --compare
```

### Python API

You can also use the project as a Python package:

```python
from src.core.genetic_algorithm import run_genetic_algorithm
from src.core.utils import print_best_individual, plot_fitness_history

# Run genetic algorithm
best_individual, best_fitness, fitness_history = run_genetic_algorithm(
    population_size=10,
    num_generations=10,
    mutation_rate=0.1,
    epochs_per_eval=5,
    batch_size=64
)

# Print results
print_best_individual(best_individual, best_fitness)

# Plot fitness history
plot_fitness_history(fitness_history)
```

For optimized (parallel) execution:

```python
from src.experiments.optimized_algorithm import run_genetic_algorithm as run_optimized

best_individual, best_fitness, fitness_history = run_optimized(
    population_size=10,
    num_generations=10,
    mutation_rate=0.1,
    epochs_per_eval=5,
    batch_size=64,
    max_workers=4
)
```

## 🎛️ Hyperparameter Space

The genetic algorithm searches through the following hyperparameter space:

| Hyperparameter                 | Values                   |
|--------------------------------|--------------------------|
| Number of convolutional layers | 1, 2, 3, 4, 5            |
| Number of filters per layer    | 16, 32, 64, 128, 256     |
| Kernel sizes                   | 3, 5, 7                  |
| Pooling types                  | max, avg, none           |
| Learning rates                 | 0.1, 0.01, 0.001, 0.0001 |
| Activation functions           | relu, elu, leaky_relu    |
| Dropout rates                  | 0.0, 0.25, 0.5           |

## 🧬 Genetic Algorithm Components

### Individual Representation

Each individual represents a CNN architecture, encoded as a dictionary of hyperparameters:

```python
{
    'conv_layers': 3,                 # Number of convolutional layers
    'learning_rate': 0.001,           # Learning rate for Adam optimizer
    'filters_0': 64,                  # Number of filters in layer 0
    'kernel_size_0': 3,               # Kernel size in layer 0
    'activation_0': 'relu',           # Activation function in layer 0
    'pool_type_0': 'max',             # Pooling type in layer 0
    'dropout_0': 0.25,                # Dropout rate in layer 0
    # ... parameters for layers 1, 2, etc.
}
```

### Genetic Operators

1. **Selection**: Tournament selection chooses individuals for reproduction based on their fitness
2. **Crossover**: Uniform crossover combines hyperparameters from two parents to create offspring
3. **Mutation**: Parameter-specific mutation randomly changes hyperparameters with varying probabilities
4. **Elitism**: The best individuals from each generation are preserved unchanged

### Fitness Evaluation

Each individual's fitness is determined by:
1. Building a CNN with the specified hyperparameters
2. Training the model on the CIFAR-10 training set
3. Evaluating the model on the validation set
4. Using the validation accuracy as the fitness score

## 📊 Experimental Results

The genetic algorithm consistently outperforms both random search and grid search, finding CNN architectures with higher validation accuracy while evaluating the same number of models.

### Performance Comparison

| Method            | Best Validation Accuracy | Evaluations | Time per Evaluation (s) |
|-------------------|--------------------------|-------------|-------------------------|
| Genetic Algorithm | 0.72                     | 40          | 18.5                    |
| Random Search     | 0.67                     | 40          | 17.8                    |
| Grid Search       | 0.65                     | 40          | 17.9                    |

### Best CNN Architecture

The best architecture discovered by the genetic algorithm has 3 convolutional layers:

- **Layer 1**: 64 filters, 3×3 kernel, ReLU activation, max pooling, 0.25 dropout
- **Layer 2**: 128 filters, 3×3 kernel, ReLU activation, max pooling, 0.25 dropout
- **Layer 3**: 64 filters, 5×5 kernel, ELU activation, average pooling, 0.5 dropout
- **Learning rate**: 0.001

This architecture achieved 72% validation accuracy and 69% test accuracy.

For detailed results, see the [final report](docs/report.md) and the [analysis notebook](notebooks/cnn_genetic_tuning_analysis.ipynb).

## ⚡ Performance Optimization

The optimized implementation includes:

- **Parallel Fitness Evaluation**: Uses multiple processes to evaluate individuals simultaneously
- **Caching System**: Avoids redundant model evaluations for identical individuals
- **Early Stopping**: Terminates unproductive training early to save time
- **Adaptive Training**: Employs changing batch sizes for more efficient training
- **Mixed Precision**: Uses reduced precision where appropriate for faster computation

These optimizations result in significant speedup compared to the original implementation.

## 🔬 Ablation Studies

I conducted ablation studies to understand the impact of different components and parameters:

### Selection Strategies

Tournament selection performed best, followed by rank selection and roulette-wheel selection.

### Crossover Strategies

Uniform crossover outperformed single-point crossover and arithmetic recombination.

### Mutation Strategies

Adaptive mutation performed best, followed by Gaussian mutation and standard mutation.

### Other Parameters

- **Population Size**: Larger populations (15–20) found better solutions but required more computation
- **Mutation Rate**: Optimal rates were in the 0.1–0.2 range
- **Tournament Size**: A tournament size of 3 offered the best balance of selection pressure and diversity

## 🔍 Visualization

The project includes tools for visualizing:

- **CNN Architectures**: Diagram representation of model architecture
- **Filters and Feature Maps**: Visualizations of what the CNN learns
- **Fitness Evolution**: Plots of fitness improvement over generations
- **Population Diversity**: Analysis of how the population evolves
- **Hyperparameter Impact**: Analysis of how different hyperparameters affect performance

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`python -m tests.run_tests`)
5. Commit your changes (`git commit -m 'Add your feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

Please ensure your code follows the project's style and includes appropriate tests and documentation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.