@echo off
echo Creating conda environment from environment.yml...
call conda env create -f environment.yml
echo Environment 'cnn-genetic' created. Activate with 'conda activate cnn-genetic'

echo Creating directory structure...
mkdir data docs\figures logs notebooks results\figures results\models results\baseline src\utils

echo Downloading CIFAR-10 dataset...
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"

echo Installing package in development mode...
pip install -e .

echo Setup complete! Run 'python main.py --help' to see available commands.