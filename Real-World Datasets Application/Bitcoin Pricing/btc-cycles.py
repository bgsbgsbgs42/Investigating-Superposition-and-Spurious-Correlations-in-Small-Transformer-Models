def analyze_market_cycles():
    # Get model predictions
    with torch.no_grad():
        predictions = model(features)
    
    # Convert to numpy for analysis
    preds_np = predictions.numpy()
    features_np = features.numpy()
    
    # Analyze cycle dependence
    cycle_metrics = {
        'yearly': np.corrcoef(features_np[:, :, 5].mean(axis=1), preds_np)[0,1],
        'monthly': np.corrcoef(features_np[:, :, 6].mean(axis=1), preds_np)[0,1],
        'quarterly': np.corrcoef(features_np[:, :, 7].mean(axis=1), preds_np)[0,1]
    }
    
    # Analyze technical vs cycle importance
    technical_importance = np.mean([
        np.abs(np.corrcoef(features_np[:, :, i].mean(axis=1), preds_np)[0,1])
        for i in range(5)
    ])
    
    cycle_importance = np.mean([
        np.abs(np.corrcoef(features_np[:, :, i].mean(axis=1), preds_np)[0,1])
        for i in range(5, 9)
    ])
    
    return {
        'cycle_correlations': cycle_metrics,
        'feature_importance': {
            'technical': technical_importance,
            'market_cycle': cycle_importance
        }
    }

cycle_analysis = analyze_market_cycles()

print("\nMarket Cycle Analysis:")
print("\nCycle Correlations:")
for cycle, corr in cycle_analysis['cycle_correlations'].items():
    print(f"{cycle}: {corr:.3f}")

print("\nFeature Importance:")
for feature_type, importance in cycle_analysis['feature_importance'].items():
    print(f"{feature_type}: {importance:.3f}")