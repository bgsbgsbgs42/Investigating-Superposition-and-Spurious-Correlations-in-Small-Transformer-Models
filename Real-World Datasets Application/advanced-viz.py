import matplotlib.pyplot as plt
import seaborn as sns

def visualize_advanced_metrics(info_flow, compression, robustness):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Information Flow
    layers = list(info_flow.keys())
    info_content = [m['info_content'] for m in info_flow.values()]
    info_flow_vals = [m['info_flow'] for m in info_flow.values()]
    
    axes[0,0].plot(info_content, 'b-', label='Information Content')
    axes[0,0].plot(info_flow_vals, 'r--', label='Information Flow')
    axes[0,0].set_xticks(range(len(layers)))
    axes[0,0].set_xticklabels(layers, rotation=45)
    axes[0,0].set_title('Information Analysis')
    axes[0,0].legend()
    
    # Compression Metrics
    eff_ranks = [m['effective_rank'] for m in compression.values()]
    comp_ratios = [m['compression_ratio'] for m in compression.values()]
    
    ax2 = axes[0,1].twinx()
    axes[0,1].bar(range(len(layers)), eff_ranks, color='b', alpha=0.5, label='Effective Rank')
    ax2.plot(range(len(layers)), comp_ratios, 'r-', label='Compression Ratio')
    axes[0,1].set_xticks(range(len(layers)))
    axes[0,1].set_xticklabels(layers, rotation=45)
    axes[0,1].set_title('Compression Analysis')
    axes[0,1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Feature Importance Stability
    sns.histplot(robustness['importance_stability'], ax=axes[1,0])
    axes[1,0].set_title('Feature Importance Stability')
    
    # Prediction Confidence
    if robustness['boundary_metrics']:
        metrics = [
            robustness['boundary_metrics']['boundary_width'],
            robustness['boundary_metrics']['local_linearity'],
            robustness['confidence']['mean_confidence']
        ]
        labels = ['Boundary Width', 'Local Linearity', 'Mean Confidence']
        axes[1,1].bar(labels, metrics)
        axes[1,1].set_title('Robustness Metrics')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

fig = visualize_advanced_metrics(info_flow, compression, robustness)