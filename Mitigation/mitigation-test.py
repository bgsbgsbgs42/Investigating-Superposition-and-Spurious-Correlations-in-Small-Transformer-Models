def evaluate_mitigation_strategies(
    features: torch.Tensor,
    labels: torch.Tensor,
    feature_groups: Dict[str, slice]
):
    strategies = MitigationStrategies(model)
    detector = EnhancedDetector(model)
    
    results = {}
    
    # Test each strategy
    for strategy in [
        strategies.adversarial_training,
        strategies.gradient_surgery,
        strategies.contrastive_regularization,
        strategies.uncertainty_weighting
    ]:
        strategy_name = strategy.__name__
        print(f"\nTesting {strategy_name}")
        
        # Apply strategy
        model_copy = copy.deepcopy(model)
        strategy(features, labels, feature_groups)
        
        # Evaluate
        with torch.no_grad():
            predictions = model_copy(features)
            accuracy = ((predictions > 0.5) == labels).float().mean()
            
            # Check spurious correlations
            spurious_scores = detector.detect_spurious_correlations(
                features, labels, feature_groups
            )
            
            # Test generalization
            permuted_idx = torch.randperm(len(features))
            test_acc = ((model_copy(features[permuted_idx]) > 0.5) == 
                       labels[permuted_idx]).float().mean()
            
        results[strategy_name] = {
            'accuracy': accuracy.item(),
            'generalization': test_acc.item(),
            'spurious_scores': spurious_scores
        }
        
    return results

# Run evaluation
results = evaluate_mitigation_strategies(features_tensor, labels_tensor, feature_groups)

print("\nStrategy Comparison:")
for strategy, metrics in results.items():
    print(f"\n{strategy}:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Generalization: {metrics['generalization']:.3f}")
    for group, scores in metrics['spurious_scores'].items():
        print(f"{group} spurious correlation: {np.mean(list(scores.values())):.3f}")