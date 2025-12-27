from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os

app = Flask(__name__)

# Load model and scaler
print("Loading model...")
model = tf.keras.models.load_model('models/magadh.keras')
scaler = joblib.load('models/magadh_scaler.pkl')

with open('models/magadh_features.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

with open('models/magadh_metrics.json', 'r') as f:
    metrics = json.load(f)

print("Model loaded successfully!")

@app.route('/')
def home():
    return jsonify({
        'message': 'Magadh Churn Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'POST - Make prediction',
            '/metrics': 'GET - Model metrics'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/metrics')
def get_metrics():
    return jsonify(metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        # Check for required features
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing features: {list(missing_features)}',
                'required_features': feature_names
            }), 400
        
        # Reorder columns to match training
        df = df[feature_names]
        
        # Scale and predict
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled, verbose=0)
        
        results = []
        for i, prob in enumerate(predictions):
            prob_value = float(prob[0])
            results.append({
                'customer_id': i,
                'churn_probability': prob_value,
                'will_churn': bool(prob_value > 0.5),
                'risk_level': 'High' if prob_value > 0.7 else 'Medium' if prob_value > 0.4 else 'Low'
            })
        
        return jsonify({
            'predictions': results,
            'model_accuracy': metrics['test_accuracy']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
