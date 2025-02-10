import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class FinancialDataProcessor:
    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        self.scaler = StandardScaler()
        
    def create_features(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        # Get historical data
        data = yf.download(symbol, start="2020-01-01", end="2024-01-01")
        
        # Technical indicators (genuine features)
        data['SMA'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['VOL'] = data['Volume'].rolling(window=20).std()
        
        # Calendar effects (potentially spurious)
        data['DayOfWeek'] = data.index.dayofweek
        data['MonthEnd'] = data.index.is_month_end.astype(int)
        
        # Create sequences
        X, y = self._create_sequences(data)
        
        return X, y
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = ['SMA', 'RSI', 'VOL', 'DayOfWeek', 'MonthEnd']
        X, y = [], []
        
        for i in range(self.lookback, len(data)):
            X.append(data[features].iloc[i-self.lookback:i].values)
            # Binary label: 1 if price increases
            y.append(data['Close'].iloc[i] > data['Close'].iloc[i-1])
            
        return np.array(X), np.array(y)

class RealWorldTransformer(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], n_heads: int = 4):
        super().__init__()
        seq_len, n_features = input_shape
        
        self.feature_embedding = nn.Linear(n_features, 64)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, 64))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=n_heads,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.output = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_embedding(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return torch.sigmoid(self.output(x)).squeeze(-1)

class RealWorldAnalyzer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.feature_groups = {
            'technical': slice(0, 3),  # SMA, RSI, VOL
            'calendar': slice(3, 5)    # DayOfWeek, MonthEnd
        }
        
    def analyze_feature_importance(self, 
                                 features: torch.Tensor,
                                 labels: torch.Tensor) -> dict:
        results = {}
        
        for group_name, group_slice in self.feature_groups.items():
            # Zero out feature group
            masked_features = features.clone()
            masked_features[:, :, group_slice] = 0
            
            # Measure impact
            with torch.no_grad():
                orig_pred = self.model(features)
                masked_pred = self.model(masked_features)
                
                # Calculate metrics
                orig_acc = ((orig_pred > 0.5) == labels).float().mean()
                masked_acc = ((masked_pred > 0.5) == labels).float().mean()
                
                results[group_name] = {
                    'importance': (orig_acc - masked_acc).item(),
                    'standalone_acc': masked_acc.item()
                }
                
        return results
    
    def temporal_stability(self,
                         features: torch.Tensor,
                         labels: torch.Tensor,
                         window_size: int = 100) -> dict:
        """Analyze how feature importance changes over time"""
        n_windows = len(features) // window_size
        temporal_results = {group: [] for group in self.feature_groups}
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            
            window_results = self.analyze_feature_importance(
                features[start_idx:end_idx],
                labels[start_idx:end_idx]
            )
            
            for group, metrics in window_results.items():
                temporal_results[group].append(metrics['importance'])
                
        return temporal_results

# Training utilities
def train_real_world_model(model: nn.Module,
                          features: torch.Tensor,
                          labels: torch.Tensor,
                          val_split: float = 0.2) -> Tuple[List[float], List[float]]:
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    # Split data
    split_idx = int(len(features) * (1 - val_split))
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    train_losses, val_losses = [], []
    
    for epoch in range(50):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_loss = criterion(val_outputs, val_labels)
            val_losses.append(val_loss.item())
            
    return train_losses, val_losses