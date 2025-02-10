import torch
import numpy as np
from typing import List, Dict
from sklearn.metrics import roc_auc_score

class ExtendedCausalAnalysis:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.cached_activations = {}
        
    def _setup_activation_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.cached_activations[name] = output
            return hook
            
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(hook_fn(name))
                
    def analyze_path_specific_effects(self,
                                    features: torch.Tensor,
                                    target_feature: int,
                                    mediator_layer: str) -> Dict[str, float]:
        """Analyze causal effects through specific paths in the network"""
        self._setup_activation_hooks()
        
        # Get baseline activations
        baseline_output = self.model(features)
        baseline_mediator = self.cached_activations[mediator_layer]
        
        # Intervene on input feature
        modified_features = features.clone()
        modified_features[:, target_feature] = torch.zeros_like(features[:, target_feature])
        
        # Direct effect (through non-mediator paths)
        direct_output = self.model(modified_features)
        direct_effect = torch.mean(torch.abs(baseline_output - direct_output))
        
        # Indirect effect (through mediator)
        self.model(modified_features)  # Update activations
        modified_mediator = self.cached_activations[mediator_layer]
        
        mediator_effect = torch.mean(torch.abs(baseline_mediator - modified_mediator))
        
        return {
            'direct_effect': direct_effect.item(),
            'mediator_effect': mediator_effect.item()
        }
        
    def robustness_analysis(self,
                           features: torch.Tensor,
                           labels: torch.Tensor,
                           noise_levels: List[float] = [0.1, 0.2, 0.5]) -> Dict[str, List[float]]:
        """Test model robustness under different types of interventions"""
        results = {
            'gaussian_noise': [],
            'feature_dropout': [],
            'adversarial': []
        }
        
        for noise in noise_levels:
            # Gaussian noise intervention
            noisy_features = features + torch.randn_like(features) * noise
            noisy_output = self.model(noisy_features)
            noisy_auc = roc_auc_score(labels.numpy(), noisy_output.detach().numpy())
            results['gaussian_noise'].append(noisy_auc)
            
            # Feature dropout intervention
            dropout_mask = torch.bernoulli(torch.ones_like(features) * (1 - noise))
            dropout_features = features * dropout_mask
            dropout_output = self.model(dropout_features)
            dropout_auc = roc_auc_score(labels.numpy(), dropout_output.detach().numpy())
            results['feature_dropout'].append(dropout_auc)
            
            # Simple adversarial intervention
            perturbed_features = features.clone().requires_grad_()
            output = self.model(perturbed_features)
            loss = torch.nn.functional.binary_cross_entropy(output, 1 - labels)
            loss.backward()
            
            with torch.no_grad():
                adversarial_features = features + noise * torch.sign(perturbed_features.grad)
                adversarial_output = self.model(adversarial_features)
                adversarial_auc = roc_auc_score(labels.numpy(), adversarial_output.numpy())
                results['adversarial'].append(adversarial_auc)
                
        return results
        
    def feature_interaction_analysis(self,
                                   features: torch.Tensor,
                                   feature_dims: List[int]) -> np.ndarray:
        """Analyze causal interactions between feature groups"""
        n_features = len(feature_dims)
        interaction_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                # Baseline prediction
                baseline_output = self.model(features)
                
                # Intervene on feature i
                modified_i = features.clone()
                start_i = sum(feature_dims[:i])
                modified_i[:, start_i:start_i+feature_dims[i]] = 0
                output_i = self.model(modified_i)
                
                # Intervene on feature j
                modified_j = features.clone()
                start_j = sum(feature_dims[:j])
                modified_j[:, start_j:start_j+feature_dims[j]] = 0
                output_j = self.model(modified_j)
                
                # Intervene on both
                modified_both = modified_i.clone()
                modified_both[:, start_j:start_j+feature_dims[j]] = 0
                output_both = self.model(modified_both)
                
                # Calculate interaction strength
                individual_effect = torch.abs(baseline_output - output_i).mean() + \
                                  torch.abs(baseline_output - output_j).mean()
                joint_effect = torch.abs(baseline_output - output_both).mean()
                
                # Interaction is difference between joint and sum of individual effects
                interaction = (joint_effect - individual_effect).item()
                
                interaction_matrix[i, j] = interaction
                interaction_matrix[j, i] = interaction
                
        return interaction_matrix

# Test the extended experiments
def run_extended_experiments(model, features, labels, feature_dims):
    analyzer = ExtendedCausalAnalysis(model)
    
    # Path-specific effects
    path_effects = analyzer.analyze_path_specific_effects(
        torch.FloatTensor(features),
        target_feature=0,  # First feature group
        mediator_layer='transformer_layers.0'
    )
    
    # Robustness analysis
    robustness_results = analyzer.robustness_analysis(
        torch.FloatTensor(features),
        torch.FloatTensor(labels)
    )
    
    # Feature interactions
    interaction_matrix = analyzer.feature_interaction_analysis(
        torch.FloatTensor(features),
        feature_dims
    )
    
    return path_effects, robustness_results, interaction_matrix

#These experiments add:

#Path-specific effect analysis
#Model robustness testing
#Feature interaction analysis