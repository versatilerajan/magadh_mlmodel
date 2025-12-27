import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

def generate_realistic_churn_data(n_samples=5000):
    """Generate realistic telecom churn data"""
    np.random.seed(42)
    
    tenure_months = np.random.exponential(scale=24, size=n_samples).clip(1, 72)
    monthly_charges = np.random.gamma(shape=2, scale=35, size=n_samples).clip(15, 150)
    total_charges = tenure_months * monthly_charges * np.random.uniform(0.9, 1.1, n_samples)
    call_duration_avg = np.random.gamma(shape=3, scale=15, size=n_samples)
    data_usage_gb = np.random.lognormal(mean=2.5, sigma=1.2, size=n_samples).clip(0, 100)
    support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    late_payments = np.random.poisson(lam=0.8, size=n_samples)
    contract_length = np.random.choice([1, 12, 24], size=n_samples, p=[0.4, 0.35, 0.25])
    num_services = np.random.randint(1, 6, size=n_samples)
    satisfaction_score = np.random.beta(a=5, b=2, size=n_samples) * 10
    competitor_offers = np.random.binomial(n=3, p=0.3, size=n_samples)
    price_sensitivity = np.random.uniform(0, 1, size=n_samples)
    
    churn_prob = (
        -0.02 * tenure_months + 0.008 * monthly_charges + 0.15 * support_tickets +
        0.20 * late_payments + -0.05 * satisfaction_score + 0.10 * competitor_offers +
        0.15 * price_sensitivity + -0.03 * num_services + -0.01 * (contract_length == 24)
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    churn = (np.random.random(n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame({
        'tenure_months': tenure_months, 'monthly_charges': monthly_charges,
        'total_charges': total_charges, 'call_duration_avg': call_duration_avg,
        'data_usage_gb': data_usage_gb, 'support_tickets': support_tickets,
        'late_payments': late_payments, 'contract_length_months': contract_length,
        'num_services': num_services, 'satisfaction_score': satisfaction_score,
        'competitor_offers': competitor_offers, 'price_sensitivity': price_sensitivity,
        'churn': churn
    })
    return df

def build_model(input_dim):
    """Build neural network model"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def main():
    print("=" * 60)
    print("MAGADH CHURN PREDICTION MODEL - TRAINING")
    print("=" * 60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Generate data
    print("\n[1/6] Generating data...")
    df = generate_realistic_churn_data(n_samples=5000)
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    # Prepare data
    print("\n[2/6] Preprocessing data...")
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model
    print("\n[3/6] Building model...")
    model = build_model(input_dim=X_train_scaled.shape[1])
    print(f"Model parameters: {model.count_params():,}")
    
    # Train model
    print("\n[4/6] Training model...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n[5/6] Evaluating model...")
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print("\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC:      {test_auc:.4f}")
    print(f"  ROC-AUC:  {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Save artifacts
    print("\n[6/6] Saving model and artifacts...")
    model.save('models/magadh.keras')
    model.save('models/magadh.h5')
    joblib.dump(scaler, 'models/magadh_scaler.pkl')
    
    with open('models/magadh_features.txt', 'w') as f:
        f.write('\n'.join(X.columns))
    
    metrics = {
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'test_roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'training_epochs': len(history.history['loss']),
        'features': list(X.columns)
    }
    
    with open('models/magadh_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("Saved files:")
    print("  - models/magadh.keras")
    print("  - models/magadh.h5")
    print("  - models/magadh_scaler.pkl")
    print("  - models/magadh_features.txt")
    print("  - models/magadh_metrics.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
