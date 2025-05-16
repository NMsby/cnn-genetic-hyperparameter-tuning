# CNN Hyperparameter Tuning with Genetic Algorithms

This project implements a genetic algorithm framework for optimizing hyperparameters of Convolutional Neural Networks
(CNNs) on the CIFAR-10 dataset.
Traditional hyperparameter tuning methods like grid search and random search can be computationally expensive
and scale poorly with the number of hyperparameters.
Genetic algorithms provide a biologically inspired approach
that can more effectively navigate complex hyperparameter spaces.

## Project Structure

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
│   └── __init__.py         # Make src a package
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

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cnn-genetic-hyperparameter-tuning.git
   cd cnn-genetic-hyperparameter-tuning
   ```

2. Create a conda environment and install dependencies:
   ```
   conda env create -f environment.yml
   conda activate cnn-genetic
   ```

   Alternatively, you can use pip:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

## Usage

### Command-line Interface

Run the genetic algorithm:
```
python main.py ga --population 10 --generations 10
```

Run the optimized genetic algorithm with parallel processing:
```
python main.py optimized --population 10 --generations 10 --workers 4
```

Run baseline methods (random search, grid search):
```
python main.py baseline --method all --samples 20
```

Run ablation studies:
```
python main.py ablation --parallel
```

Run visualizations:
```
python main.py visualize --type all
```

Run benchmarks:
```
python main.py benchmark --compare
```

### Python API

```python
from src.core.genetic_algorithm import run_genetic_algorithm
from src.core.utils import print_best_individual, plot_fitness_history

# Run genetic algorithm
best_individual, best_fitness, fitness_history = run_genetic_algorithm(
    population_size=10,
    num_generations=10,
    mutation_rate=0.1,
    epochs_per_eval=5
)

# Print results
print_best_individual(best_individual, best_fitness)

# Plot fitness history
plot_fitness_history(fitness_history)
```

## Genetic Algorithm Components

1. **Representation**: Each individual represents a set of CNN hyperparameters.
2. **Initialization**: Create an initial population of random CNN architectures.
3. **Fitness Evaluation**: Train and evaluate each CNN architecture on the validation set.
4. **Selection**: Choose the fittest individuals to become parents for the next generation.
5. **Crossover**: Combine hyperparameters from two parents to create offspring.
6. **Mutation**: Randomly change some hyperparameters to maintain diversity.
7. **Replacement**: Form a new generation from the offspring and possibly some parents.

## Hyperparameter Search Space

| Hyperparameter                 | Values                   |
|--------------------------------|--------------------------|
| Number of convolutional layers | 1, 2, 3, 4, 5            |
| Number of filters per layer    | 16, 32, 64, 128, 256     |
| Kernel sizes                   | 3, 5, 7                  |
| Pooling types                  | max, avg, none           |
| Learning rates                 | 0.1, 0.01, 0.001, 0.0001 |
| Activation functions           | relu, elu, leaky_relu    |
| Dropout rates                  | 0.0, 0.25, 0.5           |

## Results

Our genetic algorithm consistently outperforms both random search and grid search,
finding CNN architectures with higher validation accuracy while evaluating the same number of models.
The best architecture typically has 3–4 convolutional layers with a learning rate of 0.001.

For detailed results,
see the [final report](docs/report.md) and the [analysis notebook](notebooks/cnn_genetic_tuning_analysis.ipynb).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.