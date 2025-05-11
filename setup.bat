@echo off
echo Creating conda environment from environment.yml...
call conda env create -f environment.yml
echo Environment 'cnn-genetic' created. Activate with 'conda activate cnn-genetic'

echo Downloading CIFAR-10 dataset...
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"

echo Setup complete!