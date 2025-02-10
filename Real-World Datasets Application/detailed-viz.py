import matplotlib.pyplot as plt
import seaborn as sns

def plot_detailed_analysis(attention_bias, feature_interactions, temporal_stability, outcome_bias):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Attention Bias
    attention_data = pd.DataFrame({
        'Entropy': [attention_bias['clinical_entropy'], attention_bias['admin_entropy']],
        'Bias': [attention_bias['clinical_bias'], attention_bias['admin_bias']]
    }, index=['Clinical', 'Administrative'])
    attention_data.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Attention Analysis')
    
    # Feature Interactions
    interaction_data = pd.Series(feature_interactions)
    interaction_data.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Feature Interactions')
    
    # Temporal Stability
    if isinstance(temporal_stability['importance_stability'], np.ndarray):
        sns.heatmap(
            temporal_stability['importance_stability'].reshape(-1, 1),
            ax=axes[1,0],
            cmap='YlOrRd'
        )
        axes[1,0].set_title('Feature Importance Stability')
    
    # Outcome Bias
    outcome_data = pd.DataFrame(outcome_bias).T
    outcome_data['auc'].plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('AUC by Insurance Type')
    
    plt.tight_layout()
    return fig

fig = plot_detailed_analysis(attention_bias, feature_interactions, temporal_stability, outcome_bias)