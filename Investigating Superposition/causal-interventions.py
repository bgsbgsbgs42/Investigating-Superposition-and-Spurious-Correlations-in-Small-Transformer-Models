import torch
import numpy as np
from typing import List, Tuple, Dict

class CausalInterventionAnalyzer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.interventions = {}
        
    def register_intervention_point(self, name: str, module: torch.nn.Module):
        self.interventions[name] = module
        
    def counterfactual_intervention(self, 
                                  features: torch.Tensor,
                                  intervention_point: str,
                                  intervention_fn) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform counterfactual intervention at specified point"""
        original_output = self.model(features)
        
        # Store original parameters
        original_params = {}
        if intervention_point in self.interventions:
            module = self.interventions[intervention_point]
            original_params = {name: param.clone() for name, param in module.named_parameters()}
            
            # Apply intervention
            intervention_fn(module)
            
            # Get counterfactual output
            counterfactual_output = self.model(features)
            
            # Restore original parameters
            with torch.no_grad():
                for name, param in module.named_parameters():
                    param.copy_(original_params[name])
                    
            return original_output, counterfactual_output
        else:
            raise ValueError(f"Intervention point {intervention_point} not registered")

    def analyze_feature_importance(self,
                                 features: torch.Tensor,
                                 intervention_point: str,
                                 feature_dims: List[int]) -> Dict[str, float]:
        """Analyze causal importance of different feature groups"""
        importances = {}
        start_idx = 0
        
        for i, dim in enumerate(feature_dims):
            def intervention_fn(module):
                with torch.no_grad():
                    if isinstance(module, torch.nn.Linear):
                        # Zero out weights corresponding to feature
                        module.weight[:, start_idx:start_idx+dim] = 0
                        
            orig_out, cf_out = self.counterfactual_intervention(
                features, 
                intervention_point,
                intervention_fn
            )
            
            # Compute importance as output change
            importance = torch.nn.functional.mse_loss(orig_out, cf_out)
            importances[f'feature_{i}'] = importance.item()
            start_idx += dim
            
        return importances

    def test_spurious_correlation(self,
                                features: torch.Tensor,
                                labels: torch.Tensor,
                                spurious_feature_idx: int,
                                intervention_point: str) -> float:
        """Test model's reliance on spurious feature"""
        def intervention_fn(module):
            with torch.no_grad():
                if isinstance(module, torch.nn.Linear):
                    # Zero out spurious feature
                    module.weight[:, spurious_feature_idx] = 0
                    
        orig_out, cf_out = self.counterfactual_intervention(
            features,
            intervention_point,
            intervention_fn
        )
        
        # Compare accuracy with and without spurious feature
        orig_acc = ((orig_out > 0.5) == labels).float().mean()
        cf_acc = ((cf_out > 0.5) == labels).float().mean()
        
        return (orig_acc - cf_acc).item()

# Example usage and testing
def run_causal_experiments(model, features, labels, feature_dims):
    analyzer = CausalInterventionAnalyzer(model)
    
    # Register intervention points
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            analyzer.register_intervention_point(name, module)
    
    # Analyze feature importance
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.FloatTensor(labels)
    
    importances = analyzer.analyze_feature_importance(
        features_tensor,
        'input_proj',
        feature_dims
    )
    
    # Test spurious correlation
    spurious_impact = analyzer.test_spurious_correlation(
        features_tensor,
        labels_tensor,
        spurious_feature_idx=-feature_dims[-1],  # Last feature group
        intervention_point='input_proj'
    )
    
    return importances, spurious_impact

#This implementation provides:

#Counterfactual interventions to test feature importance
#Analysis of spurious correlations through causal interventions
#Quantification of model reliance on biased features