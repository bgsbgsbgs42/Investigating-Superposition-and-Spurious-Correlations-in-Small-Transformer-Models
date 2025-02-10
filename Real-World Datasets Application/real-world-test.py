def run_real_world_analysis(symbol: str = "SPY"):
    # Process data
    processor = FinancialDataProcessor()
    features, labels = processor.create_features(symbol)
    
    # Convert to tensors
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    
    # Create and train model
    model = RealWorldTransformer(input_shape=features.shape[1:])
    train_losses, val_losses = train_real_world_model(model, features, labels)
    
    # Analyze results
    analyzer = RealWorldAnalyzer(model)
    importance_results = analyzer.analyze_feature_importance(features, labels)
    temporal_results = analyzer.temporal_stability(features, labels)
    
    return {
        'importance': importance_results,
        'temporal': temporal_results,
        'training': {'train_loss': train_losses, 'val_loss': val_losses}
    }

# Run analysis
results = run_real_world_analysis()

print("\nFeature Group Importance:")
for group, metrics in results['importance'].items():
    print(f"{group}: {metrics['importance']:.3f} (standalone acc: {metrics['standalone_acc']:.3f})")

print("\nTemporal Stability:")
for group, values in results['temporal'].items():
    stability = np.std(values)
    print(f"{group} stability (std): {stability:.3f}")