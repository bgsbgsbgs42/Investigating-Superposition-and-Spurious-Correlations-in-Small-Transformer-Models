from healthcare_analysis import HealthcareTransformer, generate_synthetic_ehr, HealthcareDataProcessor

# Generate test data
data = generate_synthetic_ehr()
processor = HealthcareDataProcessor()
features, labels = processor.process_ehr_data(data)

# Setup model and feature groups
features_tensor = torch.FloatTensor(features)
labels_tensor = torch.FloatTensor(labels)
model = HealthcareTransformer(input_shape=features.shape[1:])

feature_groups = {
    'clinical': slice(0, 7),
    'administrative': slice(7, 12)
}

# Test detection and mitigation
test_detection_system(model, features_tensor, labels_tensor, feature_groups)

# Evaluate model performance before and after mitigation
def evaluate_performance(features, labels):
    with torch.no_grad():
        predictions = model(features)
        accuracy = ((predictions > 0.5) == labels).float().mean()
    return accuracy.item()

initial_accuracy = evaluate_performance(features_tensor, labels_tensor)
print(f"\nInitial Accuracy: {initial_accuracy:.3f}")

# Apply mitigation
detector = AutomatedDetector(model)
mitigator = AutomatedMitigator(model, detector)
mitigator.mitigate_spurious_correlations(features_tensor, labels_tensor, feature_groups)
mitigator.mitigate_superposition(features_tensor)

final_accuracy = evaluate_performance(features_tensor, labels_tensor)
print(f"Final Accuracy: {final_accuracy:.3f}")