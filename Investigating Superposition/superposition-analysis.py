import torch
import numpy as np
from research_utils import generate_multifeature_data
from small_transformer import SmallTransformer, train_model
from pca_analysis import SuperpositionAnalyzer

# Analysis parameters
n_features_list = [3, 5, 7]  # Test different feature counts
hidden_dims = [32, 64, 128]  # Test different model capacities
n_samples = 1000
feature_dim = 10

def analyze_capacity_vs_superposition(n_features_list, hidden_dims):
    results = {}
    
    for n_features in n_features_list:
        for hidden_dim in hidden_dims:
            # Generate data
            features, labels = generate_multifeature_data(
                n_samples=n_samples, 
                n_features=n_features, 
                feature_dim=feature_dim
            )
            
            # Create and train model
            input_dim = n_features * feature_dim
            model = SmallTransformer(input_dim=input_dim, hidden_dim=hidden_dim)
            model = train_model(model, features, labels)
            
            # Analyze superposition
            analyzer = SuperpositionAnalyzer(model)
            features_tensor = torch.FloatTensor(features)
            
            # Get overlap metrics
            feature_dims = [feature_dim] * n_features
            overlap_matrix = analyzer.analyze_feature_overlap(
                features_tensor,
                'input_proj',
                feature_dims
            )
            
            # Calculate key metrics
            avg_overlap = np.mean(overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)])
            max_overlap = np.max(overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)])
            
            # Get PCA explained variance
            _, pca = analyzer.analyze_superposition(features_tensor, 'input_proj')
            variance_explained = pca.explained_variance_ratio_.cumsum()
            
            results[(n_features, hidden_dim)] = {
                'avg_overlap': avg_overlap,
                'max_overlap': max_overlap,
                'variance_explained': variance_explained[:5]  # First 5 components
            }
    
    return results

# Run analysis
results = analyze_capacity_vs_superposition(n_features_list, hidden_dims)

# Print findings
for (n_features, hidden_dim), metrics in results.items():
    print(f"\nModel: {n_features} features, {hidden_dim} hidden dim")
    print(f"Average feature overlap: {metrics['avg_overlap']:.3f}")
    print(f"Maximum feature overlap: {metrics['max_overlap']:.3f}")
    print(f"Cumulative variance explained by first 5 PCs: {metrics['variance_explained']}")

# Additional analysis of activation patterns
def analyze_activation_patterns(n_features=5, hidden_dim=64):
    # Generate data with specific activation patterns
    features, labels = generate_multifeature_data(
        n_samples=n_samples, 
        n_features=n_features, 
        feature_dim=feature_dim
    )
    
    # Create model
    input_dim = n_features * feature_dim
    model = SmallTransformer(input_dim=input_dim, hidden_dim=hidden_dim)
    model = train_model(model, features, labels)
    
    # Analyze neuron specialization
    analyzer = SuperpositionAnalyzer(model)
    features_tensor = torch.FloatTensor(features)
    
    # Collect activations
    activations = analyzer.collect_activations(features_tensor)
    layer_activations = activations['input_proj']
    
    # Analyze neuron specialization
    neuron_stats = {
        'mean_activation': np.mean(layer_activations, axis=0),
        'std_activation': np.std(layer_activations, axis=0),
        'sparsity': np.mean(layer_activations == 0, axis=0)
    }
    
    return neuron_stats

# Run activation pattern analysis
activation_patterns = analyze_activation_patterns()
print("\nNeuron Activation Patterns:")
print(f"Mean activation range: [{np.min(activation_patterns['mean_activation']):.3f}, {np.max(activation_patterns['mean_activation']):.3f}]")
print(f"Std deviation range: [{np.min(activation_patterns['std_activation']):.3f}, {np.max(activation_patterns['std_activation']):.3f}]")
print(f"Average sparsity: {np.mean(activation_patterns['sparsity']):.3f}")

#This analysis reveals:

#How feature overlap changes with model capacity
#The distribution of information across neurons
#Sparsity patterns in neuron activations