from research_utils import generate_biased_data

# Setup
n_features = 5
feature_dim = 10
features, labels = generate_biased_data(n_samples=1000, n_features=n_features, feature_dim=feature_dim)

# Train model
model = SmallTransformer(input_dim=features.shape[1])
model = train_model(model, features, labels)

# Run experiments
feature_dims = [feature_dim] * (n_features + 1)
path_effects, robustness_results, interaction_matrix = run_extended_experiments(
    model, features, labels, feature_dims
)

print("\nPath-Specific Effects:")
for effect_type, value in path_effects.items():
    print(f"{effect_type}: {value:.3f}")

print("\nRobustness Results:")
for intervention_type, aucs in robustness_results.items():
    print(f"{intervention_type} - AUCs: {[f'{auc:.3f}' for auc in aucs]}")

print("\nFeature Interaction Matrix:")
print(interaction_matrix)