# CNN Hyperparameter Tuning with Genetic Algorithms

This repository implements a genetic algorithm approach to optimizing Convolutional Neural Network (CNN) hyperparameters on the CIFAR-10 dataset.

## Project Overview

Traditional hyperparameter tuning methods like grid search or random search can be computationally expensive and may not efficiently explore the search space. This project employs genetic algorithms—a biologically-inspired approach—to more effectively navigate complex hyperparameter landscapes.

## Learning Objectives

- Implement and understand genetic algorithm principles for neural network optimization
- Apply genetic algorithms for hyperparameter tuning of CNNs
- Evaluate and compare performance against traditional tuning methods
- Analyze the evolutionary trajectory of CNN architectures
- Visualize and interpret results from the optimization process

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Other dependencies listed in requirements.txt

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cnn-genetic-hyperparameter-tuning.git
   cd cnn-genetic-hyperparameter-tuning
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the initial test:
   ```
   python src/genetic_algorithms_starter.py
   ```

## Repository Structure

- `src/` - Source code including genetic algorithm implementation
- `tests/` - Unit tests for the implementation
- `notebooks/` - Jupyter notebooks for analysis and visualization
- `results/` - Saved models, metrics, and visualizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
