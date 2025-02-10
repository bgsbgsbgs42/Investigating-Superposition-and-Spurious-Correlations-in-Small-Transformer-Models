import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

class HealthcareDataProcessor:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.label_encoders = {}
        
    def preprocess_data(self):
        # Clinical features
        clinical_features = [
            'Blood Pressure', 'BMI', 'Blood Glucose', 'Heart Rate',
            'Physical Activity'
        ]
        
        # Administrative/demographic features (potentially spurious)
        admin_features = [
            'Gender', 'Age', 'Occupation', 'Sleep Duration',
            'Quality of Sleep'
        ]
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Occupation']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            self.data[col] = self.label_encoders[col].fit_transform(self.data[col])
        
        # Scale numerical features
        scaler = StandardScaler()
        self.data[clinical_features + admin_features] = scaler.fit_transform(
            self.data[clinical_features + admin_features]
        )
        
        # Create outcome variable (1 if stress level > median)
        median_stress = self.data['Stress Level'].median()
        y = (self.data['Stress Level'] > median_stress).astype(int)
        
        # Prepare feature matrix
        X = self.data[clinical_features + admin_features].values
        
        return torch.FloatTensor(X), torch.FloatTensor(y)

def analyze_healthcare_patterns():
    processor = HealthcareDataProcessor(path[0])  # path[0] contains the CSV file
    features, labels = processor.preprocess_data()
    
    # Initialize model
    model = LargeTransformer(input_dim=features.shape[1])
    
    # Train model
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(features.unsqueeze(1))  # Add sequence dimension
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Analyze using our framework
    analyzer = ScaleAnalyzer(model)
    detector = EnhancedDetector(model)
    
    # Split features into clinical and administrative
    feature_groups = {
        'clinical': slice(0, 5),
        'administrative': slice(5, 10)
    }
    
    # Get analysis results
    superposition_metrics = analyzer.analyze_layer_superposition(
        features.unsqueeze(1)
    )
    
    spurious_scores = detector.detect_spurious_correlations(
        features.unsqueeze(1),
        labels,
        feature_groups
    )
    
    return {
        'superposition': superposition_metrics,
        'spurious_correlations': spurious_scores,
        'model': model,
        'features': features,
        'labels': labels
    }

# Run analysis
results = analyze_healthcare_patterns()

print("\nSuperposition Analysis:")
for layer, metrics in results['superposition'].items():
    print(f"\n{layer}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

print("\nSpurious Correlation Detection:")
for group, metrics in results['spurious_correlations'].items():
    print(f"\n{group}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")