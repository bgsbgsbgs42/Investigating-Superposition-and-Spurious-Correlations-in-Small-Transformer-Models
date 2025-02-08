# Investigating-Superposition-and-Spurious-Correlations-in-Small-Transformer-Models

Exploring the open problems in mechanistic interpretability of (1) the phenomenon of superposition in small transformer models, and (2) the emergence and encoding of spurious correlations in simple neural networks. 

Findings include: 
  1. Superposition Analysis
  - Detected increasing superposition with model depth, plateauing after 8-12 layers
  - Found stronger feature interactions in deeper layers
  - Identified compression patterns in representation space
  
  2. Spurious Correlation Detection
  - Implemented automated detection system with multiple metrics:
    - Counterfactual impact
    - Distribution shift
    - Temporal consistency
    - Causal strength
  - Found administrative features showed higher spurious correlations than clinical features
  
  3. Mitigation Strategies
  - Adversarial training: Best for high-confidence spurious correlations
  - Gradient surgery: Most effective at preserving task performance
  - Contrastive regularization: Strongest feature disentanglement
  - Uncertainty weighting: Most stable decision boundaries
  
  4. Architecture Analysis
  - Parallel architecture: Better feature disentanglement but higher sensitivity
  - Hierarchical model: Strong feature compression and robustness
  - Gated architecture: Best control of spurious correlations
  
  5. Component-Level Analysis
  - Attention mechanisms showed increasing specialization with depth
  - Feed-forward networks exhibited varying degrees of sparsity
  - Component interactions revealed architecture-specific patterns
  
  6. Scaling Behaviour
  - Width scaling showed diminishing returns after certain model sizes
  - Depth scaling increased feature interactions and layer specialization
  - Gradient magnitude decreased with depth while pattern diversity increased

These findings suggest optimal architectures should combine hierarchical structure for robustness with gating mechanisms for controlling spurious correlations.

Please note this project was completed with the help of Claude.ai
