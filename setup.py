# setup.py
from setuptools import setup, find_packages

setup(
    name="cnn-genetic-tuning",
    version="0.1.0",
    description="CNN Hyperparameter Tuning with Genetic Algorithms",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "graphviz==0.20.3",
        "jupyterlab==4.4.2",
        "matplotlib==3.10.3",
        "numpy==2.1.3",
        "pandas==2.2.3",
        "pydot==4.0.0",
        "scikit-learn==1.6.1",
        "seaborn==0.13.2",
        "tensorflow==2.19.0",
        "tensorflow-datasets==4.9.8",
        "tqdm==4.67.1",
    ],
    entry_points={
        "console_scripts": [
            "cnn-genetic-tuning=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Machine Learning | Artificial  Intelligence",
    ],
)