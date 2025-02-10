from research_utils import generate_multifeature_data
from pca_analysis import SuperpositionAnalyzer

# Generate test data
features, labels = generate_multifeature_data(n_samples=1000, n_features=5, feature_dim=10)

# Train models
input_dim = features.shape[1]
standard_model = SmallTransformer(input_dim=input_dim)
disentangled_model = DisentangledTransformer(input_dim=input_dim)

standard_model = train_model(standard_model, features, labels)
disentangled_model = train_disentangled_model(disentangled_model, features, labels)

# Compare feature overlap
features_tensor = torch.FloatTensor(features)
feature_dims = [10] * 5

# Analyze standard model
std_analyzer = SuperpositionAnalyzer(standard_model)
std_overlap = std_analyzer.analyze_feature_overlap(features_tensor, 'input_proj', feature_dims)

# Analyze disentangled model
dis_analyzer = SuperpositionAnalyzer(disentangled_model)
dis_overlap = dis_analyzer.analyze_feature_overlap(features_tensor, 'input_proj', feature_dims)

print("\nFeature Overlap Comparison:")
print(f"Standard Model - Avg Overlap: {np.mean(std_overlap):.3f}")
print(f"Disentangled Model - Avg Overlap: {np.mean(dis_overlap):.3f}")