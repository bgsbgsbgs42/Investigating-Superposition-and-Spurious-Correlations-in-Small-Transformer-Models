# Run scaling analysis
analyzer = ScalingAnalyzer(model)

# Analyze width scaling
width_results = analyzer.analyze_representation_scaling(features_tensor)

# Analyze depth scaling
depth_results = analyzer.analyze_depth_scaling(features_tensor)

def summarize_scaling_trends(width_results, depth_results):
    trends = {
        'width_scaling': {
            'capacity_utilization': [],
            'feature_interactions': [],
            'representation_entropy': []
        },
        'depth_scaling': {
            'gradient_magnitude': [],
            'layer_specialization': [],
            'pattern_diversity': []
        }
    }
    
    # Analyze width scaling
    for dim, metrics in width_results.items():
        # Average capacity utilization
        capacity_util = np.mean([
            m['entropy'] for m in metrics['capacity'].values()
        ])
        trends['width_scaling']['capacity_utilization'].append(capacity_util)
        
        # Feature interactions
        trends['width_scaling']['feature_interactions'].append(
            metrics['interactions']['mean_interaction']
        )
        
        # Representation entropy
        avg_entropy = np.mean([
            layer['interference'] for layer in metrics['layer_metrics'].values()
        ])
        trends['width_scaling']['representation_entropy'].append(avg_entropy)
    
    # Analyze depth scaling
    for n_layers, metrics in depth_results.items():
        # Gradient flow
        grad_mag = np.mean([
            g['magnitude'] for g in metrics['gradient_metrics'].values()
        ])
        trends['depth_scaling']['gradient_magnitude'].append(grad_mag)
        
        # Layer specialization
        spec = np.mean([
            s['selectivity'] for s in metrics['specialization'].values()
        ])
        trends['depth_scaling']['layer_specialization'].append(spec)
        
        # Pattern diversity
        div = np.mean([
            s['pattern_diversity'] for s in metrics['specialization'].values()
        ])
        trends['depth_scaling']['pattern_diversity'].append(div)
    
    return trends

scaling_trends = summarize_scaling_trends(width_results, depth_results)

print("\nWidth Scaling Trends:")
for metric, values in scaling_trends['width_scaling'].items():
    slope = np.polyfit(range(len(values)), values, 1)[0]
    print(f"{metric}: {slope:.3f} per step")

print("\nDepth Scaling Trends:")
for metric, values in scaling_trends['depth_scaling'].items():
    slope = np.polyfit(range(len(values)), values, 1)[0]
    print(f"{metric}: {slope:.3f} per step")