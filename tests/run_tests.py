import unittest
from tests import test_genetic_algorithm

# Create test loader
loader = unittest.TestLoader()

# Create test suite from test classes
suite = unittest.TestSuite()

# Add test classes to suite
suite.addTest(loader.loadTestsFromTestCase(test_genetic_algorithm.TestGeneticAlgorithmComponents))
suite.addTest(loader.loadTestsFromTestCase(test_genetic_algorithm.TestGeneticAlgorithmIntegration))
suite.addTest(loader.loadTestsFromTestCase(test_genetic_algorithm.TestHyperparameterSearchSpace))

# Create test runner
runner = unittest.TextTestRunner(verbosity=2)

# Run tests
result = runner.run(suite)

# Print summary
print(f"\nTest Results: {result.testsRun} tests run")
print(f"  Failures: {len(result.failures)}")
print(f"  Errors: {len(result.errors)}")
print(f"  Skipped: {len(result.skipped)}")

# Exit with appropriate status code for CI integration
if result.wasSuccessful():
    print("All tests passed!")
    exit(0)
else:
    print("Some tests failed!")
    exit(1)
