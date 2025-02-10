from research_utils import generate_biased_data

# Generate biased dataset
n_features = 5
feature_dim = 10
features, labels = generate_biased_data(
    n_samples=1000,
    n_features=n_features,
    feature_dim=feature_dim,
    bias_strength=0.8
)

# Train model
input_dim = features.shape[1]
model = SmallTransformer(input_dim=input_dim)
model = train_model(model, features, labels)

# Run causal experiments
feature_dims = [feature_dim] * (n_features + 1)  # +1 for bias feature
importances, spurious_impact = run_causal_experiments(
    model,
    features,
    labels,
    feature_dims
)

print("\nFeature Importance Analysis:")
for feature, importance in importances.items():
    print(f"{feature}: {importance:.3f}")

print(f"\nSpurious Feature Impact: {spurious_impact:.3f}")
# Higher impact indicates stronger reliance on spurious correlation