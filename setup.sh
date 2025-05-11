#!/bin/bash

# Create and activate conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml
echo "Environment 'cnn-genetic' created. Activate with 'conda activate cnn-genetic'"

# Download CIFAR-10 dataset in advance to avoid download during model training
echo "Downloading CIFAR-10 dataset..."
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"

echo "Setup complete!"