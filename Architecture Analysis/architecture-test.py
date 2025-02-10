def compare_architectures(features, labels):
    variants = TransformerVariants
    input_dim = features.shape[-1]
    
    architectures = {
        'parallel': variants.ParallelTransformer(input_dim),
        'hierarchical': variants.HierarchicalTransformer(input_dim),
        'gated': variants.GatedTransformer(input_dim)
    }
    
    analyzer = ArchitectureAnalyzer(features, labels)
    results = {}
    
    for name, model in architectures.items():
        print(f"\nAnalyzing {name} architecture...")
        metrics = analyzer.analyze_architecture(model)
        results[name] = metrics
        
    return results

results = compare_architectures(features_tensor, labels_tensor)

# Analyze results
print("\nArchitecture Comparison:")
for arch, metrics in results.items():
    print(f"\n{arch.upper()}:")
    print(f"Average Rank: {np.mean([m['rank'] for m in metrics['representation'].values()]):.2f}")
    print(f"Feature Attribution Variance: {np.var(list(metrics['attribution'].values())):.3f}")
    print(f"Noise Sensitivity: {metrics['robustness']['noise_sensitivity']:.3f}")
    print(f"Feature Sensitivity: {metrics['robustness']['feature_sensitivity']:.3f}")