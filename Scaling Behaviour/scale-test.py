# Test configurations
input_dims = [256, 512, 1024]
n_layers_list = [4, 8, 12]

# Run analysis
results = analyze_scale_effects(
    input_dims,
    n_layers_list,
    features_tensor,
    labels_tensor
)

# Analyze trends
def analyze_trends(results):
    trends = {
        'superposition_by_depth': [],
        'attention_diversity': [],
        'effective_rank_ratio': []
    }
    
    for config, metrics in results.items():
        # Average superposition across layers
        superposition = np.mean([
            layer['interference']
            for layer in metrics['superposition'].values()
        ])
        trends['superposition_by_depth'].append(superposition)
        
        # Average attention diversity
        attention_div = np.mean([
            attn['head_diversity']
            for attn in metrics['attention'].values()
        ])
        trends['attention_diversity'].append(attention_div)
        
        # Effective rank ratio
        ranks = [layer['effective_rank'] for layer in metrics['superposition'].values()]
        trends['effective_rank_ratio'].append(np.mean(ranks) / max(ranks))
        
    return trends

trends = analyze_trends(results)

print("\nScaling Trends:")
for metric, values in trends.items():
    print(f"\n{metric}:")
    print(f"Min: {min(values):.3f}")
    print(f"Max: {max(values):.3f}")
    print(f"Trend: {np.polyfit(range(len(values)), values, 1)[0]:.3f} per step")