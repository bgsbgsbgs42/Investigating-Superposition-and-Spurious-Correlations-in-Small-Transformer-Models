import torch
import torch.nn as nn

class SmallTransformer(nn.Module):
    def __init__(self, input_dim, n_heads=4, n_layers=2, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Add position dimension for transformer
        x = x.unsqueeze(1)
        
        # Project input
        x = self.input_proj(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Pool and project to output
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return torch.sigmoid(x).squeeze(-1)

def train_model(model, features, labels, n_epochs=100, batch_size=32):
    """Train model with early stopping"""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    # Convert to tensors
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        # Batch training
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / (len(features) // batch_size)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    return model