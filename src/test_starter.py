import genetic_algorithms_starter as gas
import tensorflow as tf
import numpy as np
import random
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def test_data_loading():
    """Test if data loading function works properly."""
    print("Testing data loading...")
    try:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = gas.load_data()
        print(f"✓ Data loaded successfully.")
        print(f"  Training set: {x_train.shape}, {y_train.shape}")
        print(f"  Validation set: {x_val.shape}, {y_val.shape}")
        print(f"  Test set: {x_test.shape}, {y_test.shape}")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_generate_individual():
    """Test if generate_individual function works properly."""
    print("Testing generate_individual...")
    try:
        individual = gas.generate_individual()
        print(f"✓ Individual generated successfully: {individual}")
        return True
    except Exception as e:
        print(f"✗ Individual generation failed: {e}")
        return False


def test_build_model():
    """Test if build_model function works properly."""
    print("Testing build_model...")
    try:
        # Create a simple valid individual
        individual = {
            'conv_layers': 2,
            'learning_rate': 0.001,
            'filters_0': 32,
            'kernel_size_0': 3,
            'activation_0': 'relu',
            'pool_type_0': 'max',
            'dropout_0': 0.0,
            'filters_1': 64,
            'kernel_size_1': 3,
            'activation_1': 'relu',
            'pool_type_1': 'max',
            'dropout_1': 0.0
        }

        model = gas.build_model(individual)
        print(f"✓ Model built successfully.")
        print(f"  Model summary:")
        model.summary()
        return True
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False


def test_small_run():
    """Test a small run of the genetic algorithm with minimal computation."""
    print("Testing a small run of the genetic algorithm...")
    try:
        # Override the evaluate_fitness function to avoid actual training
        original_evaluate = gas.evaluate_fitness

        # Mock function to avoid actual training
        def mock_evaluate_fitness(individual, x_train, y_train, x_val, y_val, epochs=1, batch_size=64):
            time.sleep(0.1)  # Simulate some computation
            return random.random()  # Return random fitness

        # Replace it with mock
        gas.evaluate_fitness = mock_evaluate_fitness

        # Run with very small values
        try:
            best_individual, best_fitness, fitness_history = gas.run_genetic_algorithm(
                population_size=4,
                num_generations=2,
                mutation_rate=0.1,
                epochs_per_eval=1
            )
            print(f"✓ Small genetic algorithm run completed.")
            print(f"  Best fitness: {best_fitness}")
            print(f"  Fitness history: {fitness_history}")
            success = True
        finally:
            # Restore original function
            gas.evaluate_fitness = original_evaluate

        return success
    except Exception as e:
        print(f"✗ Small genetic algorithm run failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing starter code functionality...")
    tests = [
        test_data_loading,
        test_generate_individual,
        test_build_model,
        test_small_run
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    print("Test Summary:")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")

    if all(results):
        print("✓ All tests passed! The starter code is functional.")
    else:
        print("✗ Some tests failed. Check the output above for details.")
