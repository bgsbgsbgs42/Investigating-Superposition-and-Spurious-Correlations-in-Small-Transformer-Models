import torch
from research_utils import generate_multifeature_data
from small_transformer import SmallTransformer, train_model
from pca_analysis import SuperpositionAnalyzer

# Generate synthetic data
n_features = 5
feature_dim = 10
features, labels = generate_multifeature_data(n_samples=1000, n_features=n_features, feature_dim=feature_dim)

# Create and train model
input_dim = n_features * feature_dim
model = SmallTransformer(input_dim=input_dim)
model = train_model(model, features, labels)

# Analyze superposition
analyzer = SuperpositionAnalyzer(model)
features_tensor = torch.FloatTensor(features)

# Visualize activations
feature_dims = [feature_dim] * n_features
feature_names = [f'Feature {i+1}' for i in range(n_features)]

# Generate both visualizations
activation_fig = analyzer.visualize_superposition(
    features_tensor, 
    'input_proj',
    feature_names
)

overlap_fig = analyzer.plot_feature_overlap(
    features_tensor,
    'input_proj',
    feature_dims,
    feature_names
)