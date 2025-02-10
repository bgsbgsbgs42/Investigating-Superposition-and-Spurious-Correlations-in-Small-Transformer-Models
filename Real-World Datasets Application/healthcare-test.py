def generate_synthetic_ehr(n_patients=1000, n_timesteps=20):
    data = pd.DataFrame()
    
    # Generate clinical features
    data['heart_rate'] = np.random.normal(80, 10, n_patients * n_timesteps)
    data['blood_pressure'] = np.random.normal(120, 20, n_patients * n_timesteps)
    data['temperature'] = np.random.normal(37, 0.5, n_patients * n_timesteps)
    data['respiratory_rate'] = np.random.normal(16, 4, n_patients * n_timesteps)
    data['oxygen_saturation'] = np.random.normal(98, 2, n_patients * n_timesteps)
    data['lab_values'] = np.random.normal(0, 1, n_patients * n_timesteps)
    data['medications'] = np.random.randint(0, 5, n_patients * n_timesteps)
    
    # Generate administrative features with bias
    data['admission_type'] = np.random.randint(0, 3, n_patients * n_timesteps)
    data['insurance_type'] = np.random.randint(0, 4, n_patients * n_timesteps)
    data['facility_type'] = np.random.randint(0, 2, n_patients * n_timesteps)
    data['admission_day'] = np.repeat(np.arange(n_timesteps), n_patients)
    data['length_of_stay'] = np.random.randint(1, 30, n_patients * n_timesteps)
    
    # Generate outcome with both clinical and spurious correlations
    clinical_factor = (
        (data['heart_rate'] > 100) * 0.3 +
        (data['oxygen_saturation'] < 95) * 0.4 +
        (data['temperature'] > 38) * 0.3
    )
    
    spurious_factor = (
        (data['insurance_type'] == 0) * 0.2 +
        (data['admission_day'] % 7 == 0) * 0.1
    )
    
    data['outcome'] = (clinical_factor + spurious_factor > 0.5).astype(float)
    
    return data

# Generate and process data
data = generate_synthetic_ehr()
processor = HealthcareDataProcessor()
features, labels = processor.process_ehr_data(data)

# Convert to tensors
features_tensor = torch.FloatTensor(features)
labels_tensor = torch.FloatTensor(labels)

# Create and train model
model = HealthcareTransformer(input_shape=features.shape[1:])
metrics = train_healthcare_model(model, features_tensor, labels_tensor)

# Analyze results
analyzer = HealthcareAnalyzer(model)
attention_patterns = analyzer.analyze_attention_patterns(features_tensor)
correlations = analyzer.analyze_feature_correlations(features_tensor, labels_tensor)
spurious_patterns = analyzer.analyze_spurious_patterns(features_tensor, labels_tensor)

print("\nFeature Correlations:")
for group, corr in correlations.items():
    print(f"{group}: {corr:.3f}")

print("\nSpurious Patterns:")
print(f"Temporal correlation: {spurious_patterns['temporal_correlation']:.3f}")
print(f"Demographic bias: {spurious_patterns['demographic_bias']:.3f}")

print("\nFeature Importance Trends:")
print(f"Clinical gradient mean: {np.mean(metrics['clinical_importance']):.3f}")
print(f"Administrative gradient mean: {np.mean(metrics['admin_importance']):.3f}")