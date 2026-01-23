import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_sample_business_data(n_samples=5000):
    """Generate sample multi-domain business data for training"""
    np.random.seed(42)
    
    # Generate diverse business metrics
    departments = np.random.choice(['Sales', 'Marketing', 'HR', 'IT', 'Operations', 'Finance'], n_samples)
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples)
    
    # Numerical features
    revenue = np.random.gamma(shape=2, scale=50000, size=n_samples).clip(10000, 500000)
    expenses = revenue * np.random.uniform(0.4, 0.9, n_samples)
    profit = revenue - expenses
    employee_count = np.random.poisson(lam=50, size=n_samples).clip(5, 500)
    satisfaction = np.random.beta(a=5, b=2, size=n_samples) * 10
    productivity = np.random.gamma(shape=3, scale=20, size=n_samples).clip(10, 100)
    growth_rate = np.random.normal(5, 15, n_samples).clip(-50, 100)
    
    # Target: Performance Score (0-100)
    performance_score = (
        0.3 * (profit / revenue * 100) +
        0.2 * satisfaction +
        0.2 * productivity +
        0.15 * (growth_rate + 50) / 100 * 100 +
        0.15 * (50 - abs(50 - employee_count)) +
        np.random.normal(0, 5, n_samples)
    ).clip(0, 100)
    
    df = pd.DataFrame({
        'department': departments,
        'region': regions,
        'revenue': revenue,
        'expenses': expenses,
        'profit': profit,
        'employee_count': employee_count,
        'satisfaction_score': satisfaction,
        'productivity_index': productivity,
        'growth_rate': growth_rate,
        'performance_score': performance_score
    })
    
    return df

def train_analytics_models():
    """Train multiple models for different analytics tasks"""
    
    print("=" * 80)
    print("MAGADH BUSINESS ANALYTICS ENGINE - TRAINING")
    print("=" * 80)
    
    os.makedirs('models', exist_ok=True)
    
    # Generate training data
    print("\n[1/5] Generating synthetic business data...")
    df = generate_sample_business_data(n_samples=5000)
    print(f"Dataset shape: {df.shape}")
    print(f"Performance score range: {df['performance_score'].min():.2f} - {df['performance_score'].max():.2f}")
    
    # Prepare features
    print("\n[2/5] Preparing features...")
    
    # Separate numerical and categorical
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from features
    if 'performance_score' in numerical_cols:
        numerical_cols.remove('performance_score')
    
    # Encode categorical variables
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare feature matrix
    feature_cols = numerical_cols + [col + '_encoded' for col in categorical_cols]
    X = df_encoded[feature_cols]
    y = df_encoded['performance_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train performance prediction model
    print("\n[3/5] Training performance prediction model...")
    perf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    perf_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = perf_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Performance Model - R² Score: {r2:.4f}, RMSE: {rmse:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': perf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Train classification model for risk categorization
    print("\n[4/5] Training risk categorization model...")
    
    # Create risk labels
    y_risk = pd.cut(y, bins=[0, 40, 70, 100], labels=['High Risk', 'Medium Risk', 'Low Risk'])
    
    X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
        X, y_risk, test_size=0.2, random_state=42, stratify=y_risk
    )
    
    risk_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    risk_model.fit(scaler.fit_transform(X_train_risk), y_train_risk)
    
    # Save artifacts
    print("\n[5/5] Saving models and artifacts...")
    
    joblib.dump(perf_model, 'models/performance_model.pkl')
    joblib.dump(risk_model, 'models/risk_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    # Save metadata
    metadata = {
        'numerical_features': numerical_cols,
        'categorical_features': categorical_cols,
        'feature_columns': feature_cols,
        'performance_r2': float(r2),
        'performance_rmse': float(rmse),
        'feature_importance': feature_importance.to_dict('records'),
        'model_type': 'multi_domain_analytics',
        'version': '2.0.0'
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("Saved files:")
    print("  - models/performance_model.pkl")
    print("  - models/risk_model.pkl")
    print("  - models/scaler.pkl")
    print("  - models/label_encoders.pkl")
    print("  - models/model_metadata.json")
    print("=" * 80)

if __name__ == "__main__":
    train_analytics_models()
