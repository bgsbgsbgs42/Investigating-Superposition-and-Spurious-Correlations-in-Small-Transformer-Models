import yfinance as yf
import torch
import numpy as np
from typing import Tuple

class BTCDataProcessor:
    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        
    def get_btc_data(self) -> pd.DataFrame:
        btc = yf.download('BTC-USD', start='2016-02-10', end='2024-02-10', interval='1mo')
        
        # Technical features
        btc['SMA_3'] = btc['Close'].rolling(window=3).mean()
        btc['RSI'] = self._calculate_rsi(btc['Close'])
        btc['Volatility'] = btc['Close'].rolling(window=3).std()
        btc['Volume_MA'] = btc['Volume'].rolling(window=3).mean()
        btc['Price_Change'] = btc['Close'].pct_change()
        
        # Market cycle features (potentially spurious)
        btc['DayOfYear'] = btc.index.dayofyear
        btc['MonthEnd'] = btc.index.is_month_end.astype(int)
        btc['QuarterEnd'] = btc.index.is_quarter_end.astype(int)
        btc['YearEnd'] = btc.index.is_year_end.astype(int)
        
        return btc
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        features = [
            'SMA_3', 'RSI', 'Volatility', 'Volume_MA', 'Price_Change',
            'DayOfYear', 'MonthEnd', 'QuarterEnd', 'YearEnd'
        ]
        
        X, y = [], []
        for i in range(self.lookback, len(data)):
            feature_sequence = data[features].iloc[i-self.lookback:i].values
            X.append(feature_sequence)
            # Label: 1 if price increases in next period
            y.append(data['Close'].iloc[i] > data['Close'].iloc[i-1])
            
        return torch.FloatTensor(X), torch.FloatTensor(y)

# Process BTC data
processor = BTCDataProcessor()
btc_data = processor.get_btc_data()
features, labels = processor.prepare_training_data(btc_data)

# Initialize and train model
model = LargeTransformer(input_dim=features.shape[-1])
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss()

# Train for 100 epochs
for epoch in range(100):
    optimizer.zero_grad()
    output = model(features)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Analyze using our existing framework
analyzer = ScaleAnalyzer(model)
detector = EnhancedDetector(model)

# Get analysis results
superposition_metrics = analyzer.analyze_layer_superposition(features)
attention_patterns = analyzer.analyze_attention_patterns(features)
spurious_scores = detector.detect_spurious_correlations(
    features,
    labels,
    {'technical': slice(0, 5), 'market_cycle': slice(5, 9)}
)

print("\nSuperposition Analysis:")
for layer, metrics in superposition_metrics.items():
    print(f"\n{layer}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

print("\nSpurious Correlation Detection:")
for group, metrics in spurious_scores.items():
    print(f"\n{group}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")