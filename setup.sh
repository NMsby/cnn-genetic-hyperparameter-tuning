#!/bin/bash

# Create and activate conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml
echo "Environment 'cnn-genetic' created. Activate with 'conda activate cnn-genetic'"

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data docs/figures logs notebooks results/figures results/models results/baseline src/utils

# Download CIFAR-10 dataset in advance to avoid download during model training
echo "Downloading CIFAR-10 dataset..."
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

echo "Setup complete! Run 'python main.py --help' to see available commands"