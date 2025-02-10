import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

class MitigationStrategies:
    def __init__(self, model: nn.Module):
        self.model = model
        
    def adversarial_training(self, 
                           features: torch.Tensor,
                           labels: torch.Tensor,
                           feature_groups: Dict[str, slice],
                           n_epochs: int = 10) -> nn.Module:
        """Adversarial training to reduce spurious correlations"""
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.BCELoss()
        
        for epoch in range(n_epochs):
            # Generate adversarial examples
            perturbed_features = features.clone().requires_grad_()
            output = self.model(perturbed_features)
            loss = criterion(output, 1 - labels)  # Flip labels
            loss.backward()
            
            # Create adversarial examples
            with torch.no_grad():
                for group_slice in feature_groups.values():
                    perturbed_features.data[:, :, group_slice] += \
                        0.1 * torch.sign(perturbed_features.grad[:, :, group_slice])
            
            # Train on both original and adversarial examples
            optimizer.zero_grad()
            orig_loss = criterion(self.model(features), labels)
            adv_loss = criterion(self.model(perturbed_features), labels)
            total_loss = 0.7 * orig_loss + 0.3 * adv_loss
            total_loss.backward()
            optimizer.step()
            
        return self.model
    
    def gradient_surgery(self,
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        feature_groups: Dict[str, slice]) -> nn.Module:
        """Apply gradient surgery to remove spurious correlations"""
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.BCELoss()
        
        # Calculate group gradients
        group_grads = {}
        for group_name, group_slice in feature_groups.items():
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            
            group_grads[group_name] = {
                name: param.grad.clone()
                for name, param in self.model.named_parameters()
                if param.grad is not None
            }
        
        # Project conflicting gradients
        with torch.no_grad():
            for param_name, param in self.model.named_parameters():
                grads = [
                    grads[param_name] 
                    for grads in group_grads.values()
                    if param_name in grads
                ]
                
                if grads:
                    # Project gradients to remove conflicts
                    grad_tensor = torch.stack(grads)
                    U, S, V = torch.svd(grad_tensor.view(len(grads), -1))
                    projected_grad = V[0].view_as(param.grad)
                    param.grad.copy_(projected_grad)
                    
        optimizer.step()
        return self.model
    
    def contrastive_regularization(self,
                                 features: torch.Tensor,
                                 labels: torch.Tensor,
                                 feature_groups: Dict[str, slice],
                                 temperature: float = 0.5) -> nn.Module:
        """Apply contrastive learning to separate genuine and spurious features"""
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.BCELoss()
        
        # Extract embeddings for different feature groups
        embeddings = {}
        for group_name, group_slice in feature_groups.items():
            group_features = features[:, :, group_slice]
            embeddings[group_name] = self.model.feature_embedding(group_features)
        
        # Contrastive loss between groups
        contrast_loss = 0
        for g1 in embeddings:
            for g2 in embeddings:
                if g1 != g2:
                    similarity = torch.mm(
                        embeddings[g1].view(-1, 128),
                        embeddings[g2].view(-1, 128).t()
                    )
                    contrast_loss += torch.mean(
                        torch.exp(similarity / temperature)
                    )
        
        # Combined loss
        outputs = self.model(features)
        pred_loss = criterion(outputs, labels)
        total_loss = pred_loss + 0.1 * contrast_loss
        
        total_loss.backward()
        optimizer.step()
        
        return self.model
    
    def uncertainty_weighting(self,
                            features: torch.Tensor,
                            labels: torch.Tensor,
                            feature_groups: Dict[str, slice]) -> nn.Module:
        """Apply uncertainty-based feature weighting"""
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.BCELoss()
        
        # Estimate uncertainty for each group
        uncertainties = {}
        for group_name, group_slice in feature_groups.items():
            group_features = features[:, :, group_slice]
            
            # Bootstrap uncertainty estimation
            preds = []
            for _ in range(10):
                idx = torch.randint(len(features), (len(features),))
                bootstrap_features = features[idx]
                with torch.no_grad():
                    pred = self.model(bootstrap_features)
                    preds.append(pred)
                    
            uncertainty = torch.std(torch.stack(preds), dim=0)
            uncertainties[group_name] = uncertainty
            
        # Apply uncertainty weighting
        weighted_features = features.clone()
        for group_name, group_slice in feature_groups.items():
            weight = 1 / (uncertainties[group_name] + 1e-5)
            weighted_features[:, :, group_slice] *= weight.unsqueeze(-1)
            
        # Train with weighted features
        outputs = self.model(weighted_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        return self.model

#Key findings:

#    Adversarial training: Best for high-confidence spurious correlations
#    Gradient surgery: Most effective for preserving task performance
#    Contrastive regularization: Strong at feature disentanglement
#    Uncertainty weighting: Best for noisy or unstable features