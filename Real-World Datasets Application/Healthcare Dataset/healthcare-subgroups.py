def analyze_subgroups(results):
    features = results['features']
    labels = results['labels']
    model = results['model']
    
    # Define subgroups based on age and occupation
    age_groups = {
        'young': features[:, 6] < -0.5,
        'middle': (features[:, 6] >= -0.5) & (features[:, 6] <= 0.5),
        'older': features[:, 6] > 0.5
    }
    
    occupations = features[:, 7].unique()
    
    # Analyze predictions across subgroups
    subgroup_analysis = {}
    
    with torch.no_grad():
        predictions = model(features.unsqueeze(1))
        
        # Age group analysis
        for group_name, mask in age_groups.items():
            group_metrics = {
                'accuracy': ((predictions[mask] > 0.5) == labels[mask]).float().mean().item(),
                'bias': torch.mean(predictions[mask] - labels[mask]).item(),
                'feature_importance': analyze_feature_importance(
                    model, features[mask].unsqueeze(1), labels[mask]
                )
            }
            subgroup_analysis[f'age_{group_name}'] = group_metrics
        
        # Occupation analysis
        for occ in occupations:
            mask = features[:, 7] == occ
            occ_metrics = {
                'accuracy': ((predictions[mask] > 0.5) == labels[mask]).float().mean().item(),
                'bias': torch.mean(predictions[mask] - labels[mask]).item(),
                'feature_importance': analyze_feature_importance(
                    model, features[mask].unsqueeze(1), labels[mask]
                )
            }
            subgroup_analysis[f'occupation_{int(occ)}'] = occ_metrics
    
    return subgroup_analysis

def analyze_feature_importance(model, features, labels):
    base_pred = model(features)
    importance = []
    
    for i in range(features.shape[2]):
        modified = features.clone()
        modified[:, :, i] = 0
        importance.append(
            torch.abs(model(modified) - base_pred).mean().item()
        )
    
    return importance

# Run subgroup analysis
subgroup_results = analyze_subgroups(results)

print("\nSubgroup Analysis:")
for group, metrics in subgroup_results.items():
    print(f"\n{group}:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Bias: {metrics['bias']:.3f}")
    print(f"Most important features: {np.argsort(metrics['feature_importance'])[-3:]}")