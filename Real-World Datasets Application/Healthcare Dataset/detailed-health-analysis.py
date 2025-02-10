class HealthcareDetailedAnalyzer:
    def __init__(self, model, features, labels):
        self.model = model
        self.features = features
        self.labels = labels
        
    def analyze_stress_patterns(self):
        """Analyze patterns in stress level predictions"""
        clinical_indices = slice(0, 5)
        admin_indices = slice(5, 10)
        
        with torch.no_grad():
            base_pred = self.model(self.features.unsqueeze(1))
            
            # Clinical-only prediction
            clinical_features = self.features.clone()
            clinical_features[:, admin_indices] = 0
            clinical_pred = self.model(clinical_features.unsqueeze(1))
            
            # Admin-only prediction
            admin_features = self.features.clone()
            admin_features[:, clinical_indices] = 0
            admin_pred = self.model(admin_features.unsqueeze(1))
            
            return {
                'clinical_accuracy': ((clinical_pred > 0.5) == self.labels).float().mean().item(),
                'admin_accuracy': ((admin_pred > 0.5) == self.labels).float().mean().item(),
                'combined_accuracy': ((base_pred > 0.5) == self.labels).float().mean().item(),
                'clinical_contribution': torch.corrcoef(
                    torch.stack([base_pred.squeeze(), clinical_pred.squeeze()])
                )[0,1].item(),
                'admin_contribution': torch.corrcoef(
                    torch.stack([base_pred.squeeze(), admin_pred.squeeze()])
                )[0,1].item()
            }

    def analyze_interaction_effects(self):
        """Analyze interaction between clinical and administrative features"""
        interactions = np.zeros((5, 5))  # Clinical x Admin interactions
        
        for i in range(5):  # Clinical features
            for j in range(5):  # Admin features
                interaction = self._measure_interaction(i, j+5)
                interactions[i, j] = interaction
                
        return {
            'interaction_matrix': interactions,
            'strongest_pairs': self._get_top_interactions(interactions),
            'interaction_strength': np.mean(np.abs(interactions))
        }
    
    def _measure_interaction(self, feat1, feat2):
        modified = self.features.clone()
        
        # Measure individual and joint effects
        with torch.no_grad():
            base_pred = self.model(self.features.unsqueeze(1))
            
            modified[:, feat1] = 0
            effect1 = self.model(modified.unsqueeze(1))
            
            modified = self.features.clone()
            modified[:, feat2] = 0
            effect2 = self.model(modified.unsqueeze(1))
            
            modified[:, [feat1, feat2]] = 0
            joint_effect = self.model(modified.unsqueeze(1))
            
            interaction = torch.mean(
                torch.abs((base_pred - joint_effect) - 
                         ((base_pred - effect1) + (base_pred - effect2)))
            ).item()
            
        return interaction
    
    def _get_top_interactions(self, matrix):
        indices = np.unravel_index(
            np.argsort(matrix.ravel())[-3:],
            matrix.shape
        )
        return list(zip(indices[0], indices[1]))
    
    def analyze_temporal_stability(self):
        """Analyze prediction stability across different conditions"""
        with torch.no_grad():
            predictions = self.model(self.features.unsqueeze(1))
            
        # Analyze stability across feature ranges
        stability = {}
        for i in range(self.features.shape[1]):
            feature_values = self.features[:, i]
            quartiles = torch.quantile(feature_values, torch.tensor([0.25, 0.5, 0.75]))
            
            pred_std = []
            for j in range(3):
                if j == 0:
                    mask = feature_values <= quartiles[0]
                elif j == 1:
                    mask = (feature_values > quartiles[0]) & (feature_values <= quartiles[1])
                else:
                    mask = feature_values > quartiles[1]
                    
                pred_std.append(torch.std(predictions[mask]).item())
                
            stability[f'feature_{i}'] = {
                'prediction_std': pred_std,
                'range_sensitivity': max(pred_std) - min(pred_std)
            }
            
        return stability

# Run detailed analysis
analyzer = HealthcareDetailedAnalyzer(results['model'], results['features'], results['labels'])

stress_patterns = analyzer.analyze_stress_patterns()
interactions = analyzer.analyze_interaction_effects()
stability = analyzer.analyze_temporal_stability()

print("\nStress Pattern Analysis:")
for metric, value in stress_patterns.items():
    print(f"{metric}: {value:.3f}")

print("\nTop Feature Interactions:")
for i, j in interactions['strongest_pairs']:
    print(f"Clinical feature {i} x Admin feature {j}: {interactions['interaction_matrix'][i,j]:.3f}")

print("\nStability Analysis:")
for feature, metrics in stability.items():
    print(f"\n{feature}:")
    print(f"Range sensitivity: {metrics['range_sensitivity']:.3f}")