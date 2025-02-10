import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_financial_data(n_samples=1000, n_days=252):
    """Generate synthetic financial dataset"""
    # Date range
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    
    # Generate price data
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.lognormal(10, 1, n_days),
        # Technical features
        'SMA_20': pd.Series(prices).rolling(20).mean(),
        'RSI': pd.Series(returns).rolling(14).apply(
            lambda x: 100 - (100 / (1 + np.mean(x[x > 0]) / abs(np.mean(x[x < 0]))))
        ),
        'Volatility': pd.Series(returns).rolling(20).std(),
        # Calendar effects (potentially spurious)
        'DayOfWeek': dates.dayofweek,
        'MonthEnd': dates.is_month_end.astype(int),
        'QuarterEnd': dates.is_quarter_end.astype(int)
    })
    
    return data

def generate_healthcare_data(n_patients=1000, n_visits=10):
    """Generate synthetic healthcare dataset"""
    data = []
    patient_ids = range(n_patients)
    
    for patient_id in patient_ids:
        # Generate baseline health metrics
        baseline_hr = np.random.normal(75, 10)
        baseline_bp = np.random.normal(120, 15)
        baseline_temp = np.random.normal(37, 0.3)
        
        for visit in range(n_visits):
            # Clinical features (genuine)
            heart_rate = baseline_hr + np.random.normal(0, 5)
            blood_pressure = baseline_bp + np.random.normal(0, 8)
            temperature = baseline_temp + np.random.normal(0, 0.2)
            respiratory_rate = np.random.normal(16, 2)
            oxygen_saturation = min(100, np.random.normal(97, 2))
            lab_values = np.random.normal(0, 1)
            medications = np.random.randint(0, 5)
            
            # Administrative features (potentially spurious)
            admission_type = np.random.randint(0, 3)
            insurance_type = np.random.randint(0, 4)
            facility_type = np.random.randint(0, 2)
            admission_day = visit
            length_of_stay = np.random.randint(1, 10)
            
            # Generate outcome based on clinical factors with some spurious correlation
            clinical_risk = (
                (heart_rate > 100) * 0.3 +
                (oxygen_saturation < 95) * 0.4 +
                (temperature > 38) * 0.3
            )
            
            admin_bias = (
                (insurance_type == 0) * 0.2 +
                (admission_type == 2) * 0.15
            )
            
            outcome = int((clinical_risk + admin_bias) > 0.5)
            
            data.append({
                'patient_id': patient_id,
                'visit': visit,
                'heart_rate': heart_rate,
                'blood_pressure': blood_pressure,
                'temperature': temperature,
                'respiratory_rate': respiratory_rate,
                'oxygen_saturation': oxygen_saturation,
                'lab_values': lab_values,
                'medications': medications,
                'admission_type': admission_type,
                'insurance_type': insurance_type,
                'facility_type': facility_type,
                'admission_day': admission_day,
                'length_of_stay': length_of_stay,
                'outcome': outcome
            })
    
    return pd.DataFrame(data)

# Generate example datasets
financial_data = generate_financial_data()
healthcare_data = generate_healthcare_data()

print("\nFinancial Data Sample:")
print(financial_data.head())
print("\nHealthcare Data Sample:")
print(healthcare_data.head())