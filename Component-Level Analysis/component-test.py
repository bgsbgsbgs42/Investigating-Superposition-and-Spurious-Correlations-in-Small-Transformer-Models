def test_components():
    variants = TransformerVariants
    input_dim = features_tensor.shape[-1]
    
    architectures = {
        'parallel': variants.ParallelTransformer(input_dim),
        'hierarchical': variants.HierarchicalTransformer(input_dim),
        'gated': variants.GatedTransformer(input_dim)
    }
    
    results = compare_component_behaviors(architectures, features_tensor)
    
    component_summary = {}
    for arch_name, metrics in results.items():
        summary = {
            'attention_entropy': np.mean([
                m['focus']['entropy'] 
                for m in metrics['attention'].values()
            ]),
            'head_diversity': np.mean([
                m['specialization']['head_diversity'] 
                for m in metrics['attention'].values()
            ]),
            'ffn_sparsity': np.mean([
                m['weight_stats']['sparsity'] 
                for m in metrics['feedforward'].values()
            ]),
            'component_correlation': np.mean([
                m['correlation'] 
                for m in metrics['interactions'].values()
            ])
        }
        component_summary[arch_name] = summary
        
    return component_summary

summary = test_components()

print("\nComponent Analysis Summary:")
for arch, metrics in summary.items():
    print(f"\n{arch.upper()}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")