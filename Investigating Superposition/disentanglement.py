import torch
import torch.nn as nn
import torch.nn.functional as F

class DisentangledTransformer(nn.Module):
    def __init__(self, input_dim, n_heads=4, n_layers=2, hidden_dim=64, orthogonal_penalty=0.1):
        super().__init__()
        self.orthogonal_penalty = orthogonal_penalty
        self.input_proj = OrthogonalLinear(input_dim, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            DisentangledTransformerLayer(
                hidden_dim,
                n_heads,
                orthogonal_penalty
            ) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        
        orthogonal_loss = self.input_proj.orthogonal_loss()
        
        for layer in self.transformer_layers:
            x, layer_loss = layer(x)
            orthogonal_loss += layer_loss
            
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return torch.sigmoid(x).squeeze(-1), orthogonal_loss * self.orthogonal_penalty

class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.linear(x)
    
    def orthogonal_loss(self):
        weight = self.linear.weight
        gram_matrix = torch.mm(weight, weight.t())
        identity = torch.eye(weight.size(0), device=weight.device)
        return F.mse_loss(gram_matrix, identity)

class DisentangledTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, orthogonal_penalty):
        super().__init__()
        self.orthogonal_penalty = orthogonal_penalty
        
        # Multi-head attention with orthogonality constraint
        self.self_attn = DisentangledMultiHeadAttention(hidden_dim, n_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Feedforward with orthogonality constraint
        self.ff = nn.Sequential(
            OrthogonalLinear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            OrthogonalLinear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Self-attention
        attn_out, attn_loss = self.self_attn(x)
        x = self.norm1(x + attn_out)
        
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Compute orthogonality losses
        ff_loss = sum(layer.orthogonal_loss() for layer in self.ff if isinstance(layer, OrthogonalLinear))
        total_loss = attn_loss + ff_loss
        
        return x, total_loss

class DisentangledMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.q_proj = OrthogonalLinear(hidden_dim, hidden_dim)
        self.k_proj = OrthogonalLinear(hidden_dim, hidden_dim)
        self.v_proj = OrthogonalLinear(hidden_dim, hidden_dim)
        self.out_proj = OrthogonalLinear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_out = attn_out.view(batch_size, seq_len, hidden_dim)
        out = self.out_proj(attn_out)
        
        # Compute orthogonality loss
        orthogonal_loss = (
            self.q_proj.orthogonal_loss() +
            self.k_proj.orthogonal_loss() +
            self.v_proj.orthogonal_loss() +
            self.out_proj.orthogonal_loss()
        )
        
        return out, orthogonal_loss

def train_disentangled_model(model, features, labels, n_epochs=100, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_ortho_loss = 0
        
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs, ortho_loss = model(batch_features)
            pred_loss = criterion(outputs, batch_labels)
            
            # Combined loss
            loss = pred_loss + ortho_loss
            loss.backward()
            optimizer.step()
            
            total_loss += pred_loss.item()
            total_ortho_loss += ortho_loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Ortho Loss = {total_ortho_loss:.4f}")
    
    return model