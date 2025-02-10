class BTCDetailedAnalyzer:
    def __init__(self, model, features, labels):
        self.model = model
        self.features = features
        self.labels = labels
        
    def analyze_volatility_regimes(self):
        """Analyze model behavior in different volatility regimes"""
        volatility = self.features[:, :, 2].mean(dim=1)  # Volatility feature
        
        # Define regimes
        low_vol = torch.quantile(volatility, 0.33)
        high_vol = torch.quantile(volatility, 0.66)
        
        regimes = {
            'low_vol': volatility <= low_vol,
            'med_vol': (volatility > low_vol) & (volatility <= high_vol),
            'high_vol': volatility > high_vol
        }
        
        regime_metrics = {}
        with torch.no_grad():
            predictions = self.model(self.features)
            
            for regime_name, mask in regimes.items():
                regime_metrics[regime_name] = {
                    'accuracy': ((predictions[mask] > 0.5) == 
                               self.labels[mask]).float().mean().item(),
                    'confidence': torch.abs(predictions[mask] - 0.5).mean().item(),
                    'feature_importance': self._analyze_feature_importance(mask)
                }
                
        return regime_metrics
    
    def _analyze_feature_importance(self, mask):
        base_pred = self.model(self.features[mask])
        importance = []
        
        for i in range(self.features.shape[-1]):
            modified = self.features[mask].clone()
            modified[:, :, i] = 0
            new_pred = self.model(modified)
            importance.append(torch.abs(base_pred - new_pred).mean().item())
            
        return importance
    
    def analyze_market_phases(self):
        """Analyze model behavior in different market phases"""
        returns = self.features[:, :, 4].mean(dim=1)  # Price change feature
        
        phases = {
            'bull': returns > 0.05,
            'bear': returns < -0.05,
            'sideways': (returns >= -0.05) & (returns <= 0.05)
        }
        
        phase_analysis = {}
        for phase_name, mask in phases.items():
            # Analyze feature interactions in each phase
            phase_features = self.features[mask]
            
            if len(phase_features) > 0:
                phase_analysis[phase_name] = {
                    'feature_correlations': self._analyze_feature_correlations(phase_features),
                    'attention_patterns': self._analyze_attention_patterns(phase_features)
                }
                
        return phase_analysis
    
    def _analyze_feature_correlations(self, features):
        flat_features = features.reshape(-1, features.shape[-1])
        return torch.corrcoef(flat_features.T)
    
    def _analyze_attention_patterns(self, features):
        with torch.no_grad():
            attention_weights = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    _, weights = module(features, features, features)
                    attention_weights.append(weights.mean(dim=1))
                    
            return torch.stack(attention_weights).mean(dim=0)
    
    def analyze_temporal_dependencies(self):
        """Analyze temporal dependencies in predictions"""
        with torch.no_grad():
            predictions = self.model(self.features)
            
        # Analyze autocorrelation
        pred_autocorr = torch.tensor([
            torch.corrcoef(predictions[:-i], predictions[i:])[0,1].item()
            for i in range(1, 6)
        ])
        
        # Analyze seasonal patterns
        monthly_pattern = torch.stack([
            predictions[i::12].mean() for i in range(12)
        ])
        
        return {
            'autocorrelation': pred_autocorr,
            'monthly_pattern': monthly_pattern
        }

# Run detailed analysis
analyzer = BTCDetailedAnalyzer(model, features, labels)

volatility_analysis = analyzer.analyze_volatility_regimes()
phase_analysis = analyzer.analyze_market_phases()
temporal_analysis = analyzer.analyze_temporal_dependencies()

print("\nVolatility Regime Analysis:")
for regime, metrics in volatility_analysis.items():
    print(f"\n{regime}:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Confidence: {metrics['confidence']:.3f}")
    
print("\nMarket Phase Analysis:")
for phase, metrics in phase_analysis.items():
    print(f"\n{phase}:")
    print("Feature correlations:")
    print(metrics['feature_correlations'])
    
print("\nTemporal Dependencies:")
print(f"Autocorrelation (1-5 lags): {temporal_analysis['autocorrelation']}")
print(f"Monthly seasonality strength: {temporal_analysis['monthly_pattern'].std().item():.3f}")