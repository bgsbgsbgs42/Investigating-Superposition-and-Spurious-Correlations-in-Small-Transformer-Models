import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_multifeature_data(n_samples=1000, n_features=5, feature_dim=10):
    """Generate synthetic data with multiple independent features"""
    features = []
    labels = []
    
    for _ in range(n_samples):
        # Generate independent feature vectors
        sample_features = []
        for _ in range(n_features):
            feature = np.random.randn(feature_dim)
            sample_features.append(feature)
        
        # Create label as nonlinear combination of features
        label = np.sum([np.sum(f**2) for f in sample_features]) > n_features * feature_dim/2
        
        features.append(np.concatenate(sample_features))
        labels.append(label)
    
    return np.array(features), np.array(labels)

def generate_biased_data(n_samples=1000, n_features=5, feature_dim=10, bias_strength=0.8):
    """Generate dataset with intentional spurious correlations"""
    features, labels = generate_multifeature_data(n_samples, n_features, feature_dim)
    
    # Introduce spurious correlation
    bias_feature = np.random.randn(n_samples, feature_dim)
    bias_labels = (np.sum(bias_feature**2, axis=1) > feature_dim/2)
    
    # Mix true labels with bias
    mixed_labels = np.where(
        np.random.random(n_samples) < bias_strength,
        bias_labels,
        labels
    )
    
    # Concatenate bias feature
    biased_features = np.concatenate([features, bias_feature], axis=1)
    
    return biased_features, mixed_labels

class ActivationPatcher:
    """Tools for analyzing neuron activations"""
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
            
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_neuron_importance(self, inputs, labels, neuron_idx):
        """Measure importance of specific neurons via intervention"""
        original_output = self.model(inputs)
        
        # Zero out specific neuron
        for name, activation in self.activations.items():
            activation_copy = activation.clone()
            activation_copy[:, neuron_idx] = 0
            
            # Run forward pass with modified activation
            modified_output = self.model(inputs)
            
            # Compute importance score
            importance = F.mse_loss(original_output, modified_output)
            
        return importance.item()

def analyze_spurious_correlations(model, features, labels, feature_dims):
    """Analyze model's reliance on different feature groups"""
    importances = []
    
    for i, dim in enumerate(feature_dims):
        # Zero out feature group
        masked_features = features.clone()
        start_idx = sum(feature_dims[:i])
        end_idx = start_idx + dim
        masked_features[:, start_idx:end_idx] = 0
        
        # Measure impact on predictions
        original_preds = model(features)
        masked_preds = model(masked_features)
        importance = F.mse_loss(original_preds, masked_preds)
        importances.append(importance.item())
        
    return importances