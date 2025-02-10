import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Tuple

class SuperpositionAnalyzer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = {}
        self._setup_hooks()
        
    def _setup_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook
            
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(hook_fn(name))
    
    def collect_activations(self, features: torch.Tensor) -> dict:
        """Collect activations for input features"""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(features)
        return self.activations.copy()
    
    def analyze_superposition(self, 
                            features: torch.Tensor, 
                            layer_name: str,
                            n_components: int = 3) -> Tuple[np.ndarray, PCA]:
        """Analyze superposition in specified layer using PCA"""
        activations = self.collect_activations(features)
        layer_activations = activations[layer_name]
        
        # Reshape if needed (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
        if len(layer_activations.shape) == 3:
            layer_activations = layer_activations.reshape(-1, layer_activations.shape[-1])
            
        # Perform PCA
        pca = PCA(n_components=n_components)
        projected_activations = pca.fit_transform(layer_activations)
        
        return projected_activations, pca
    
    def visualize_superposition(self,
                              features: torch.Tensor,
                              layer_name: str,
                              feature_labels: List[str] = None):
        """Create visualization of superposition patterns"""
        projected_acts, pca = self.analyze_superposition(features, layer_name)
        
        # Create scatter plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(projected_acts[:, 0],
                           projected_acts[:, 1],
                           projected_acts[:, 2],
                           c=range(len(projected_acts)),
                           cmap='viridis')
        
        # Add labels
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} var)')
        
        if feature_labels:
            for i, label in enumerate(feature_labels):
                ax.text(projected_acts[i, 0],
                       projected_acts[i, 1],
                       projected_acts[i, 2],
                       label)
        
        plt.title(f'Neuron Activation Space: {layer_name}')
        plt.colorbar(scatter, label='Sample Index')
        
        return fig
    
    def analyze_feature_overlap(self,
                              features: torch.Tensor,
                              layer_name: str,
                              feature_dims: List[int]) -> np.ndarray:
        """Analyze how different features overlap in neuron space"""
        _, pca = self.analyze_superposition(features, layer_name)
        
        # Get principal components for each feature dimension
        overlap_matrix = np.zeros((len(feature_dims), len(feature_dims)))
        start_idx = 0
        
        for i, dim1 in enumerate(feature_dims):
            for j, dim2 in enumerate(feature_dims):
                if i <= j:
                    # Calculate overlap using cosine similarity of PC loadings
                    pc1 = pca.components_[:, start_idx:start_idx + dim1]
                    pc2 = pca.components_[:, start_idx + dim1:start_idx + dim1 + dim2]
                    
                    similarity = np.abs(np.dot(pc1.flatten(), pc2.flatten())) / \
                               (np.linalg.norm(pc1) * np.linalg.norm(pc2))
                    
                    overlap_matrix[i, j] = similarity
                    overlap_matrix[j, i] = similarity
            
            start_idx += dim1
            
        return overlap_matrix
    
    def plot_feature_overlap(self,
                           features: torch.Tensor,
                           layer_name: str, 
                           feature_dims: List[int],
                           feature_names: List[str] = None):
        """Visualize feature overlap as a heatmap"""
        overlap_matrix = self.analyze_feature_overlap(features, layer_name, feature_dims)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(overlap_matrix, cmap='YlOrRd')
        plt.colorbar(label='Feature Overlap')
        
        if feature_names:
            plt.xticks(range(len(feature_names)), feature_names, rotation=45)
            plt.yticks(range(len(feature_names)), feature_names)
        
        plt.title(f'Feature Overlap Analysis: {layer_name}')
        plt.tight_layout()
        
        return plt.gcf()