import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import seaborn as sns
import json
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
from sklearn.manifold import TSNE

# Import the necessary functions from our module
from genetic_algorithms_starter import (
    build_model,
    load_data,
    print_best_individual
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Visualization")


def visualize_filters(
        model: tf.keras.Model,
        layer_name: Optional[str] = None,
        num_filters: int = 16,
        filename: str = "filter_visualization"
):
    """
    Visualize the filters (weights) of a convolutional layer.

    Args:
        model: Trained CNN model
        layer_name: Name of the convolutional layer to visualize.
                    If None, use the first convolutional layer.
        num_filters: Number of filters to visualize
        filename: Base filename for the output image
    """
    logger.info("Visualizing CNN filters...")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Find the first convolutional layer if layer_name is not specified
    if layer_name is None:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

        if layer_name is None:
            logger.error("No convolutional layer found in the model")
            return

    # Get the layer
    try:
        layer = model.get_layer(layer_name)
    except ValueError:
        logger.error(f"Layer '{layer_name}' not found in the model")
        return

    # Check if it's a convolutional layer
    if not isinstance(layer, tf.keras.layers.Conv2D):
        logger.error(f"Layer '{layer_name}' is not a convolutional layer")
        return

    # Get the weights
    weights, biases = layer.get_weights()

    # Normalize the weights for better visualization
    weights_min, weights_max = np.min(weights), np.max(weights)
    weights = (weights - weights_min) / (weights_max - weights_min + 1e-8)

    # Get the number of filters actually in the layer
    actual_num_filters = weights.shape[3]
    num_filters = min(num_filters, actual_num_filters)

    # Create a grid to display filters
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    # Plot each filter
    for i in range(grid_size):
        for j in range(grid_size):
            filter_idx = i * grid_size + j
            if filter_idx < num_filters:
                # Get the filter
                filter_weights = weights[:, :, :, filter_idx]

                # If the filter has multiple channels (RGB), take the mean
                if filter_weights.shape[2] > 1:
                    filter_weights = np.mean(filter_weights, axis=2)
                else:
                    filter_weights = filter_weights[:, :, 0]

                # Display the filter
                axes[i, j].imshow(filter_weights, cmap='viridis')
                axes[i, j].set_title(f"Filter {filter_idx}")
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.suptitle(f"Filters from layer: {layer_name}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save the figure
    plt.savefig(f"results/{filename}.png", dpi=150, bbox_inches='tight')
    logger.info(f"Filter visualization saved to results/{filename}.png")
    plt.close()


def visualize_feature_maps(
        model: tf.keras.Model,
        image: np.ndarray,
        layer_names: Optional[List[str]] = None,
        num_feature_maps: int = 16,
        filename: str = "feature_map_visualization"
):
    """
    Visualize feature maps (activations) for a sample image.

    Args:
        model: CNN model
        image: Input image (should be preprocessed and have batch dimension)
        layer_names: List of layers to visualize. If None, visualize all Conv2D layers.
        num_feature_maps: Number of feature maps to visualize per layer
        filename: Base filename for the output images
    """
    logger.info("Visualizing feature maps...")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # Get all convolutional layers if layer_names is not specified
    if layer_names is None:
        layer_names = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_names.append(layer.name)

    # Create a list of (layer_name, layer_output) tuples
    outputs = [model.get_layer(name).output for name in layer_names]

    # Create a model that returns the activations
    activation_model = Model(inputs=model.input, outputs=outputs)

    # Get activations
    activations = activation_model.predict(image)
    if not isinstance(activations, list):
        activations = [activations]

    # Visualize activations for each layer
    for layer_idx, (layer_name, activation) in enumerate(zip(layer_names, activations)):
        # Get the number of feature maps in this layer
        actual_num_feature_maps = activation.shape[-1]
        num_to_display = min(num_feature_maps, actual_num_feature_maps)

        # Create a grid to display feature maps
        grid_size = int(np.ceil(np.sqrt(num_to_display)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

        # Plot each feature map
        for i in range(grid_size):
            for j in range(grid_size):
                feature_idx = i * grid_size + j
                if feature_idx < num_to_display:
                    # Get the feature map
                    feature_map = activation[0, :, :, feature_idx]

                    # Display the feature map
                    axes[i, j].imshow(feature_map, cmap='viridis')
                    axes[i, j].set_title(f"Feature {feature_idx}")
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')

        plt.suptitle(f"Feature Maps from layer: {layer_name}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Save the figure
        layer_filename = f"{filename}_{layer_idx}_{layer_name.replace('/', '_')}"
        plt.savefig(f"results/{layer_filename}.png", dpi=150, bbox_inches='tight')
        logger.info(f"Feature map visualization for layer {layer_name} saved to results/{layer_filename}.png")
        plt.close()


def visualize_hyperparameter_impact(
        individuals: List[Dict[str, Any]],
        fitness_scores: List[float],
        filename: str = "hyperparameter_impact"
):
    """
    Visualize the impact of different hyperparameters on model performance.

    Args:
        individuals: List of hyperparameter sets (individuals)
        fitness_scores: Corresponding fitness scores
        filename: Base filename for the output images
    """
    logger.info("Visualizing hyperparameter impact...")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Create a DataFrame for easier analysis
    data = []
    for ind, score in zip(individuals, fitness_scores):
        entry = {
            'fitness': score,
            'conv_layers': ind['conv_layers'],
            'learning_rate': ind['learning_rate']
        }

        # Add layer-specific parameters (average across layers)
        filters = []
        kernel_sizes = []
        for i in range(ind['conv_layers']):
            filters.append(ind[f'filters_{i}'])
            kernel_sizes.append(ind[f'kernel_size_{i}'])

        entry['avg_filters'] = np.mean(filters)
        entry['avg_kernel_size'] = np.mean(kernel_sizes)

        # Count pool types and activations
        pool_counts = {'max': 0, 'avg': 0, 'none': 0}
        activation_counts = {'relu': 0, 'elu': 0, 'leaky_relu': 0}

        for i in range(ind['conv_layers']):
            pool_counts[ind[f'pool_type_{i}']] += 1
            activation_counts[ind[f'activation_{i}']] += 1

        # Find the dominant pool type and activation
        entry['dominant_pool'] = max(pool_counts.items(), key=lambda x: x[1])[0]
        entry['dominant_activation'] = max(activation_counts.items(), key=lambda x: x[1])[0]

        # Average dropout rate
        dropout_rates = [ind[f'dropout_{i}'] for i in range(ind['conv_layers'])]
        entry['avg_dropout'] = np.mean(dropout_rates)

        data.append(entry)

    df = pd.DataFrame(data)

    # 1. Number of Convolutional Layers vs Fitness
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='conv_layers', y='fitness', data=df)
    plt.title('Impact of Number of Convolutional Layers on Fitness')
    plt.xlabel('Number of Convolutional Layers')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"results/{filename}_conv_layers.png", dpi=150, bbox_inches='tight')
    logger.info(f"Conv layers impact visualization saved to results/{filename}_conv_layers.png")
    plt.close()

    # 2. Learning Rate vs Fitness
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='learning_rate', y='fitness', data=df)
    plt.title('Impact of Learning Rate on Fitness')
    plt.xlabel('Learning Rate')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"results/{filename}_learning_rate.png", dpi=150, bbox_inches='tight')
    logger.info(f"Learning rate impact visualization saved to results/{filename}_learning_rate.png")
    plt.close()

    # 3. Average Filters vs Fitness
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='avg_filters', y='fitness', data=df)
    plt.title('Impact of Average Number of Filters on Fitness')
    plt.xlabel('Average Number of Filters')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add regression line
    sns.regplot(x='avg_filters', y='fitness', data=df, scatter=False, line_kws={"color": "red"})

    plt.savefig(f"results/{filename}_avg_filters.png", dpi=150, bbox_inches='tight')
    logger.info(f"Average filters impact visualization saved to results/{filename}_avg_filters.png")
    plt.close()

    # 4. Pool Type vs Fitness
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dominant_pool', y='fitness', data=df)
    plt.title('Impact of Dominant Pooling Type on Fitness')
    plt.xlabel('Dominant Pooling Type')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"results/{filename}_pool_type.png", dpi=150, bbox_inches='tight')
    logger.info(f"Pool type impact visualization saved to results/{filename}_pool_type.png")
    plt.close()

    # 5. Activation Function vs Fitness
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dominant_activation', y='fitness', data=df)
    plt.title('Impact of Dominant Activation Function on Fitness')
    plt.xlabel('Dominant Activation Function')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"results/{filename}_activation.png", dpi=150, bbox_inches='tight')
    logger.info(f"Activation function impact visualization saved to results/{filename}_activation.png")
    plt.close()

    # 6. Average Dropout Rate vs Fitness
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='avg_dropout', y='fitness', data=df)
    plt.title('Impact of Average Dropout Rate on Fitness')
    plt.xlabel('Average Dropout Rate')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add regression line
    sns.regplot(x='avg_dropout', y='fitness', data=df, scatter=False, line_kws={"color": "red"})

    plt.savefig(f"results/{filename}_avg_dropout.png", dpi=150, bbox_inches='tight')
    logger.info(f"Average dropout impact visualization saved to results/{filename}_avg_dropout.png")
    plt.close()

    # 7. Interaction Plot: Conv Layers and Learning Rate
    plt.figure(figsize=(12, 8))
    for lr in sorted(df['learning_rate'].unique()):
        subset = df[df['learning_rate'] == lr]
        means = subset.groupby('conv_layers')['fitness'].mean().reset_index()
        plt.plot(means['conv_layers'], means['fitness'], marker='o', label=f'LR = {lr}')

    plt.title('Interaction Between Number of Layers and Learning Rate')
    plt.xlabel('Number of Convolutional Layers')
    plt.ylabel('Average Fitness (Validation Accuracy)')
    plt.legend(title='Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"results/{filename}_layers_lr_interaction.png", dpi=150, bbox_inches='tight')
    logger.info(f"Layers-LR interaction visualization saved to results/{filename}_layers_lr_interaction.png")
    plt.close()

    # 8. Hyperparameter Space Visualization using t-SNE
    try:
        # Create feature matrix for t-SNE
        X = df[['conv_layers', 'avg_filters', 'avg_kernel_size', 'avg_dropout']].values

        # Add one-hot encoding for categorical variables
        pool_dummies = pd.get_dummies(df['dominant_pool'], prefix='pool')
        activation_dummies = pd.get_dummies(df['dominant_activation'], prefix='activation')
        lr_dummies = pd.get_dummies(df['learning_rate'], prefix='lr')

        # Combine all features
        X_combined = np.hstack([X, pool_dummies.values, activation_dummies.values, lr_dummies.values])

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_combined)

        # Create plot
        plt.figure(figsize=(12, 10))
        sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['fitness'], cmap='viridis',
                         s=100, alpha=0.8)

        plt.colorbar(sc, label='Fitness (Validation Accuracy)')
        plt.title('t-SNE Visualization of Hyperparameter Space')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Annotate top 5 models
        top_indices = np.argsort(df['fitness'].values)[-5:]
        for idx in top_indices:
            plt.annotate(f"Top {len(top_indices) - np.where(top_indices == idx)[0][0]}",
                         (X_tsne[idx, 0], X_tsne[idx, 1]),
                         xytext=(10, 10), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

        plt.savefig(f"results/{filename}_tsne.png", dpi=150, bbox_inches='tight')
        logger.info(f"t-SNE visualization saved to results/{filename}_tsne.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error creating t-SNE visualization: {e}")

    # 9. Correlation matrix between hyperparameters and fitness
    plt.figure(figsize=(10, 8))
    corr_data = df[['fitness', 'conv_layers', 'learning_rate', 'avg_filters',
                    'avg_kernel_size', 'avg_dropout']].copy()

    # Convert learning rate to log scale for correlation
    corr_data['log_learning_rate'] = np.log10(corr_data['learning_rate'])
    corr_data.drop('learning_rate', axis=1, inplace=True)

    # Calculate correlation matrix
    corr_matrix = corr_data.corr()

    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Hyperparameters and Fitness')
    plt.tight_layout()
    plt.savefig(f"results/{filename}_correlation.png", dpi=150, bbox_inches='tight')
    logger.info(f"Correlation matrix saved to results/{filename}_correlation.png")
    plt.close()


def visualize_evolution_progress(
        fitness_history: List[float],
        population_histories: Optional[List[List[Dict[str, Any]]]] = None,
        population_fitness_histories: Optional[List[List[float]]] = None,
        filename: str = "evolution_progress"
):
    """
    Visualize the progress of the genetic algorithm.

    Args:
        fitness_history: Best fitness in each generation
        population_histories: (Optional) Full population in each generation
        population_fitness_histories: (Optional) All fitness scores in each generation
        filename: Base filename for the output images
    """
    logger.info("Visualizing evolution progress...")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # 1. Plot fitness over generations
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, 'b-', linewidth=2, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Validation Accuracy)')
    plt.title('Best Fitness Over Generations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, len(fitness_history) + 1))

    # Add text annotations for improvement
    for i in range(1, len(fitness_history)):
        if fitness_history[i] > fitness_history[i - 1]:
            improvement = (fitness_history[i] - fitness_history[i - 1]) * 100
            plt.annotate(f"+{improvement:.2f}%",
                         (i + 1, fitness_history[i]),
                         xytext=(10, 5), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    plt.savefig(f"results/{filename}_best_fitness.png", dpi=150, bbox_inches='tight')
    logger.info(f"Best fitness visualization saved to results/{filename}_best_fitness.png")
    plt.close()

    # 2. Plot population statistics if available
    if population_fitness_histories is not None:
        plt.figure(figsize=(12, 8))

        # Calculate statistics for each generation
        gen_count = len(population_fitness_histories)
        gen_means = [np.mean(gen_fitness) for gen_fitness in population_fitness_histories]
        gen_medians = [np.median(gen_fitness) for gen_fitness in population_fitness_histories]
        gen_mins = [np.min(gen_fitness) for gen_fitness in population_fitness_histories]
        gen_maxs = [np.max(gen_fitness) for gen_fitness in population_fitness_histories]
        gen_std = [np.std(gen_fitness) for gen_fitness in population_fitness_histories]

        # Create x values
        x = np.arange(1, gen_count + 1)

        # Plot mean with error bars (standard deviation)
        plt.errorbar(x, gen_means, yerr=gen_std, fmt='o-', capsize=5,
                     label='Mean ± Std Dev', color='blue', ecolor='lightblue')

        # Plot median
        plt.plot(x, gen_medians, 's--', color='green', label='Median')

        # Plot min and max
        plt.plot(x, gen_mins, 'v:', color='red', label='Min')
        plt.plot(x, gen_maxs, '^:', color='purple', label='Max')

        # Fill between min and max
        plt.fill_between(x, gen_mins, gen_maxs, alpha=0.2, color='gray')

        plt.xlabel('Generation')
        plt.ylabel('Fitness (Validation Accuracy)')
        plt.title('Population Fitness Statistics Over Generations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(x)

        plt.savefig(f"results/{filename}_population_stats.png", dpi=150, bbox_inches='tight')
        logger.info(f"Population statistics visualization saved to results/{filename}_population_stats.png")
        plt.close()

        # 3. Plot fitness distribution per generation as violin plot
        plt.figure(figsize=(14, 8))

        # Prepare data for violin plot
        violin_data = []
        for i, gen_fitness in enumerate(population_fitness_histories):
            violin_data.extend([(i + 1, f) for f in gen_fitness])

        df_violin = pd.DataFrame(violin_data, columns=['Generation', 'Fitness'])

        # Create violin plot
        sns.violinplot(x='Generation', y='Fitness', data=df_violin, inner='box', palette='viridis')
        plt.title('Fitness Distribution per Generation')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.savefig(f"results/{filename}_fitness_distribution.png", dpi=150, bbox_inches='tight')
        logger.info(f"Fitness distribution visualization saved to results/{filename}_fitness_distribution.png")
        plt.close()

    # 3. Plot hyperparameter evolution if population histories are available
    if population_histories is not None:
        # Track evolution of number of layers
        plt.figure(figsize=(12, 6))

        # Calculate statistics for each generation
        conv_layers_per_gen = []
        for gen_population in population_histories:
            conv_layers = [ind['conv_layers'] for ind in gen_population]
            conv_layers_per_gen.append(conv_layers)

        # Create box plot
        plt.boxplot(conv_layers_per_gen, labels=range(1, len(population_histories) + 1))

        plt.xlabel('Generation')
        plt.ylabel('Number of Convolutional Layers')
        plt.title('Evolution of Number of Convolutional Layers')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.savefig(f"results/{filename}_layers_evolution.png", dpi=150, bbox_inches='tight')
        logger.info(f"Layers evolution visualization saved to results/{filename}_layers_evolution.png")
        plt.close()

        # Track evolution of learning rate (as scatter plot of frequencies)
        plt.figure(figsize=(12, 6))

        # Get unique learning rates
        all_lr = []
        for gen_population in population_histories:
            all_lr.extend([ind['learning_rate'] for ind in gen_population])
        unique_lr = sorted(set(all_lr))

        # Count learning rates per generation
        data = []
        for gen_idx, gen_population in enumerate(population_histories):
            lr_counts = {}
            for lr in unique_lr:
                lr_counts[lr] = 0

            for ind in gen_population:
                lr_counts[ind['learning_rate']] += 1

            for lr, count in lr_counts.items():
                data.append((gen_idx + 1, lr, count))

        df_lr = pd.DataFrame(data, columns=['Generation', 'Learning_Rate', 'Count'])

        # Create grouped bar chart
        for lr in unique_lr:
            subset = df_lr[df_lr['Learning_Rate'] == lr]
            plt.plot(subset['Generation'], subset['Count'], 'o-',
                     label=f'LR = {lr}', markersize=8)

        plt.xlabel('Generation')
        plt.ylabel('Count in Population')
        plt.title('Evolution of Learning Rate Distribution')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Learning Rate')
        plt.xticks(range(1, len(population_histories) + 1))

        plt.savefig(f"results/{filename}_lr_evolution.png", dpi=150, bbox_inches='tight')
        logger.info(f"Learning rate evolution visualization saved to results/{filename}_lr_evolution.png")
        plt.close()

    logger.info("Evolution progress visualization complete")


def load_and_visualize_best_model(results_file: str = "results/genetic_algorithm_results.json"):
    """
    Load the best model from results and create visualizations.

    Args:
        results_file: Path to the results JSON file
    """
    logger.info(f"Loading best model from {results_file}...")

    try:
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Extract best individual
        best_individual = results['best_individual']

        # Print details
        logger.info("Best model details:")
        print_best_individual(best_individual, results['best_fitness'])

        # Build the model
        model = build_model(best_individual)

        # Load a sample of data for feature map visualization
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
        sample_image = x_test[0]  # Use first test image as example

        # Visualize filters
        visualize_filters(model, num_filters=16, filename="best_model_filters")

        # Visualize feature maps
        visualize_feature_maps(
            model,
            sample_image,
            num_feature_maps=9,
            filename="best_model_feature_maps"
        )

        # Visualize evolution progress if available
        if 'fitness_history' in results:
            visualize_evolution_progress(
                results['fitness_history'],
                filename="best_model_evolution"
            )

        logger.info("Best model visualization complete")

    except Exception as e:
        logger.error(f"Error loading or visualizing best model: {e}")


def visualize_confusion_matrix(
        model: tf.keras.Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
        filename: str = "confusion_matrix"
):
    """
    Visualize the confusion matrix for model predictions.

    Args:
        model: Trained CNN model
        x_test: Test data
        y_test: Test labels (one-hot encoded)
        class_names: Optional list of class names
        filename: Base filename for the output image
    """
    logger.info("Visualizing confusion matrix...")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Get predictions
    y_pred = model.predict(x_test)

    # Convert from one-hot to class indices
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    conf_matrix = tf.math.confusion_matrix(y_true, y_pred).numpy()

    # Normalize by row (true labels)
    row_sums = conf_matrix.sum(axis=1)
    norm_conf_matrix = conf_matrix / row_sums[:, np.newaxis]

    # Create a figure
    plt.figure(figsize=(10, 8))

    # Plot confusion matrix
    sns.heatmap(
        norm_conf_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')

    # Save the figure
    plt.savefig(f"results/{filename}.png", dpi=150, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to results/{filename}.png")
    plt.close()


def run_visualization_suite():
    """
    Run a comprehensive suite of visualizations on existing results.
    Looks for existing result files and creates visualizations based on what's available.
    """
    logger.info("Running comprehensive visualization suite...")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # 1. Visualization of the best model from genetic algorithm
    if os.path.exists("results/genetic_algorithm_results.json"):
        logger.info("Found genetic algorithm results, visualizing best model...")
        load_and_visualize_best_model("results/genetic_algorithm_results.json")
    else:
        logger.warning("No genetic algorithm results found, skipping best model visualization")

    # 2. Hyperparameter impact visualization from random search
    if os.path.exists("results/random_search_results.json"):
        try:
            logger.info("Found random search results, visualizing hyperparameter impact...")

            with open("results/random_search_results.json", 'r') as f:
                random_results = json.load(f)

            # Extract individuals and fitness scores
            individuals = random_results.get('all_individuals', [])
            if not individuals and 'best_individual' in random_results:
                # If all individuals not saved, use what we have
                individuals = [random_results['best_individual']]

            fitness_scores = random_results.get('fitness_scores', [])
            if not fitness_scores and 'best_fitness' in random_results:
                fitness_scores = [random_results['best_fitness']]

            # Visualize hyperparameter impact if we have enough data
            if len(individuals) > 5 and len(individuals) == len(fitness_scores):
                visualize_hyperparameter_impact(
                    individuals,
                    fitness_scores,
                    filename="random_search_hyperparameter_impact"
                )
            else:
                logger.warning("Not enough data for hyperparameter impact visualization from random search")
        except Exception as e:
            logger.error(f"Error visualizing random search results: {e}")

    # 3. Comparison visualization if all methods are available
    if (os.path.exists("results/genetic_algorithm_results.json") and
            os.path.exists("results/random_search_results.json") and
            os.path.exists("results/grid_search_results.json")):
        try:
            logger.info("Found results for all methods, creating comparison...")

            # Load results
            with open("results/genetic_algorithm_results.json", 'r') as f:
                ga_results = json.load(f)

            with open("results/random_search_results.json", 'r') as f:
                random_results = json.load(f)

            with open("results/grid_search_results.json", 'r') as f:
                grid_results = json.load(f)

            # Create comparison plots
            from src import plot_comparison_results
            plot_comparison_results(
                ga_results,
                random_results,
                grid_results,
                filename="methods_comparison_visualization"
            )
        except Exception as e:
            logger.error(f"Error creating comparison visualization: {e}")

    # 4. Visualize confusion matrix for the best model
    try:
        # Load data
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

        # CIFAR-10 class names
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        # Check for the best model from genetic algorithm
        if os.path.exists("results/genetic_algorithm_results.json"):
            with open("results/genetic_algorithm_results.json", 'r') as f:
                ga_results = json.load(f)

            best_individual = ga_results['best_individual']

            # Build and train model
            logger.info("Training best model for confusion matrix visualization...")
            model = build_model(best_individual)

            # Train on combined training and validation data
            x_train_full = np.concatenate([x_train, x_val])
            y_train_full = np.concatenate([y_train, y_val])

            model.fit(
                x_train_full, y_train_full,
                epochs=5,  # Few epochs for visualization
                batch_size=64,
                verbose=1
            )

            # Create confusion matrix
            visualize_confusion_matrix(
                model,
                x_test,
                y_test,
                class_names=class_names,
                filename="best_model_confusion_matrix"
            )
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e}")

    # 5. Create t-SNE visualization of model feature space
    try:
        if os.path.exists("results/genetic_algorithm_results.json"):
            with open("results/genetic_algorithm_results.json", 'r') as f:
                ga_results = json.load(f)

            best_individual = ga_results['best_individual']

            # Build model without the classification head
            logger.info("Creating feature space visualization...")
            model = build_model(best_individual)

            # Create a feature extractor model by removing the output layer
            feature_model = Model(
                inputs=model.input,
                outputs=model.get_layer('flatten').output  # Use the flatten layer output
            )

            # Sample a subset of test data for visualization
            sample_size = 500
            indices = np.random.choice(len(x_test), sample_size, replace=False)
            x_sample = x_test[indices]
            y_sample = y_test[indices]

            # Extract features
            features = feature_model.predict(x_sample)

            # Reduce dimensionality with t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            features_tsne = tsne.fit_transform(features)

            # Plot t-SNE results
            plt.figure(figsize=(12, 10))

            # Convert one-hot to class indices
            y_indices = np.argmax(y_sample, axis=1)

            # Create scatter plot
            scatter = plt.scatter(
                features_tsne[:, 0],
                features_tsne[:, 1],
                c=y_indices,
                cmap='tab10',
                alpha=0.7,
                s=50
            )

            # Add legend
            legend = plt.legend(
                handles=scatter.legend_elements()[0],
                labels=class_names,
                title="Classes",
                loc="best"
            )

            plt.title('t-SNE Visualization of CNN Feature Space')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.grid(True, linestyle='--', alpha=0.5)

            # Save the figure
            plt.savefig("results/feature_space_tsne.png", dpi=150, bbox_inches='tight')
            logger.info("Feature space visualization saved to results/feature_space_tsne.png")
            plt.close()
    except Exception as e:
        logger.error(f"Error creating feature space visualization: {e}")

    logger.info("Visualization suite complete!")


if __name__ == "__main__":
    # Run comprehensive visualization suite
    run_visualization_suite()
