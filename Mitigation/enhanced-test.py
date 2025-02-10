def test_enhanced_system(model, features, labels, feature_groups):
    detector = EnhancedDetector(model)
    mitigator = EnhancedMitigator(model, detector)
    
    # Initial detection
    initial_scores = detector.detect_spurious_correlations(
        features, labels, feature_groups
    )
    
    print("\nInitial Detection:")
    for group, metrics in initial_scores.items():
        print(f"\n{group}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.3f}")
    
    # Apply mitigation
    mitigator.mitigate_spurious_correlations(features, labels, feature_groups)
    
    # Post-mitigation detection
    final_scores = detector.detect_spurious_correlations(
        features, labels, feature_groups
    )
    
    print("\nPost-Mitigation Detection:")
    for group, metrics in final_scores.items():
        print(f"\n{group}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.3f}")
            
    # Performance impact
    with torch.no_grad():
        initial_pred = model(features)
        initial_acc = ((initial_pred > 0.5) == labels).float().mean()
        
        # Check generalization
        permuted_idx = torch.randperm(len(features))
        test_features = features[permuted_idx]
        test_labels = labels[permuted_idx]
        
        test_pred = model(test_features)
        test_acc = ((test_pred > 0.5) == test_labels).float().mean()
        
    return {
        'initial_scores': initial_scores,
        'final_scores': final_scores,
        'accuracy': {
            'initial': initial_acc.item(),
            'generalization': test_acc.item()
        }
    }

# Run enhanced test
results = test_enhanced_system(model, features_tensor, labels_tensor, feature_groups)

print("\nAccuracy Metrics:")
print(f"Initial: {results['accuracy']['initial']:.3f}")
print(f"Generalization: {results['accuracy']['generalization']:.3f}")