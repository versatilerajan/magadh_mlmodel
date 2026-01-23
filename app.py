from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Global model variables
performance_model = None
risk_model = None
scaler = None
label_encoders = None
metadata = {}
models_loaded = False

def load_models():
    """Load models on startup"""
    global performance_model, risk_model, scaler, label_encoders, metadata, models_loaded
    
    try:
        print("=" * 60)
        print("LOADING MODELS...")
        print("=" * 60)
        
        if not os.path.exists('models'):
            print("ERROR: models directory not found")
            return False
        
        print("Models directory contents:")
        for f in os.listdir('models'):
            print(f"  - {f}")
        
        performance_model = joblib.load('models/performance_model.pkl')
        print("✓ Performance model loaded")
        
        risk_model = joblib.load('models/risk_model.pkl')
        print("✓ Risk model loaded")
        
        scaler = joblib.load('models/scaler.pkl')
        print("✓ Scaler loaded")
        
        label_encoders = joblib.load('models/label_encoders.pkl')
        print("✓ Label encoders loaded")
        
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        print("✓ Metadata loaded")
        
        print("=" * 60)
        print("ALL MODELS LOADED SUCCESSFULLY")
        print("=" * 60)
        
        models_loaded = True
        return True
        
    except Exception as e:
        print(f"ERROR loading models: {e}")
        print(traceback.format_exc())
        models_loaded = False
        return False

# Load models when module is imported
print("Initializing application...")
load_models()

def analyze_column_semantics(df):
    """Analyze column types"""
    column_types = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        if df[col].dtype in ['int64', 'float64']:
            if any(k in col_lower for k in ['revenue', 'sales', 'income']):
                column_types[col] = 'revenue'
            elif any(k in col_lower for k in ['expense', 'cost', 'spend']):
                column_types[col] = 'expense'
            elif any(k in col_lower for k in ['profit', 'margin']):
                column_types[col] = 'profit'
            elif any(k in col_lower for k in ['employee', 'staff', 'headcount']):
                column_types[col] = 'employee_count'
            elif any(k in col_lower for k in ['satisfaction', 'rating', 'score']):
                column_types[col] = 'satisfaction'
            elif any(k in col_lower for k in ['product', 'efficiency']):
                column_types[col] = 'productivity'
            elif any(k in col_lower for k in ['growth', 'increase']):
                column_types[col] = 'growth'
            else:
                column_types[col] = 'numerical_metric'
        else:
            if any(k in col_lower for k in ['dept', 'department', 'division']):
                column_types[col] = 'department'
            elif any(k in col_lower for k in ['region', 'location', 'area']):
                column_types[col] = 'region'
            else:
                column_types[col] = 'category'
    
    return column_types

def calculate_metrics(df, column_types):
    """Calculate basic metrics"""
    metrics = {
        'overview': {
            'total_records': int(len(df)),
            'total_columns': int(len(df.columns)),
            'numerical_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
            'categorical_columns': int(len(df.select_dtypes(include=['object']).columns))
        },
        'numerical_summary': {},
        'categorical_summary': {},
        'correlations': []
    }
    
    # Numerical summary
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].notna().sum() > 0:
            metrics['numerical_summary'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
    
    # Categorical summary
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].notna().sum() > 0:
            vc = df[col].value_counts()
            if len(vc) > 0:
                metrics['categorical_summary'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'most_common': str(vc.index[0]),
                    'distribution': {str(k): int(v) for k, v in vc.head(5).items()}
                }
    
    return metrics

def find_lagging_areas(df, column_types):
    """Find underperforming areas"""
    lagging = []
    
    group_cols = [c for c, t in column_types.items() if t in ['department', 'region', 'category']]
    perf_cols = [c for c, t in column_types.items() if t in ['revenue', 'profit', 'satisfaction']]
    
    for gc in group_cols[:1]:
        for pc in perf_cols[:2]:
            try:
                grouped = df.groupby(gc)[pc].mean()
                overall_mean = df[pc].mean()
                
                for name, val in grouped.items():
                    if val < overall_mean * 0.8:
                        gap_pct = ((overall_mean - val) / overall_mean) * 100
                        lagging.append({
                            'category': gc,
                            'name': str(name),
                            'metric': pc,
                            'current_value': float(val),
                            'average_value': float(overall_mean),
                            'gap_percentage': float(gap_pct),
                            'severity': 'critical' if gap_pct > 30 else 'moderate'
                        })
            except:
                continue
    
    return sorted(lagging, key=lambda x: x['gap_percentage'], reverse=True)[:5]

def find_booming_areas(df, column_types):
    """Find high-performing areas"""
    booming = []
    
    group_cols = [c for c, t in column_types.items() if t in ['department', 'region', 'category']]
    perf_cols = [c for c, t in column_types.items() if t in ['revenue', 'profit', 'satisfaction']]
    
    for gc in group_cols[:1]:
        for pc in perf_cols[:2]:
            try:
                grouped = df.groupby(gc)[pc].mean()
                overall_mean = df[pc].mean()
                
                for name, val in grouped.items():
                    if val > overall_mean * 1.2:
                        out_pct = ((val - overall_mean) / overall_mean) * 100
                        booming.append({
                            'category': gc,
                            'name': str(name),
                            'metric': pc,
                            'current_value': float(val),
                            'average_value': float(overall_mean),
                            'outperformance_percentage': float(out_pct),
                            'potential': 'excellent' if out_pct > 40 else 'good'
                        })
            except:
                continue
    
    return sorted(booming, key=lambda x: x['outperformance_percentage'], reverse=True)[:5]

def generate_insights(lagging, booming):
    """Generate actionable insights"""
    insights = {
        'critical_actions': [],
        'quick_wins': [],
        'growth_opportunities': []
    }
    
    for lag in lagging[:2]:
        if lag['severity'] == 'critical':
            insights['critical_actions'].append({
                'priority': 'HIGH',
                'area': f"{lag['category']}: {lag['name']}",
                'issue': f"{lag['gap_percentage']:.1f}% below average in {lag['metric']}",
                'recommendation': "Immediate intervention required"
            })
    
    for boom in booming[:2]:
        insights['growth_opportunities'].append({
            'priority': 'STRATEGIC',
            'area': f"{boom['category']}: {boom['name']}",
            'strength': f"{boom['outperformance_percentage']:.1f}% above average",
            'recommendation': "Scale successful strategies"
        })
    
    return insights

@app.route('/')
def home():
    return jsonify({
        'service': 'Magadh Business Analytics',
        'version': '2.0.0',
        'status': 'online' if models_loaded else 'models not loaded',
        'endpoints': {
            '/health': 'Health check',
            '/analyze': 'POST - Analyze data'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/metrics')
def get_metrics():
    return jsonify(metadata if models_loaded else {'error': 'Models not loaded'})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        # Get data
        if 'file' in request.files:
            df = pd.read_csv(request.files['file'])
        elif request.is_json:
            data = request.get_json()
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        else:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate
        if df.empty or len(df) < 5:
            return jsonify({'error': 'Insufficient data'}), 400
        
        # Analyze
        column_types = analyze_column_semantics(df)
        metrics = calculate_metrics(df, column_types)
        lagging = find_lagging_areas(df, column_types)
        booming = find_booming_areas(df, column_types)
        insights = generate_insights(lagging, booming)
        
        return jsonify({
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'dataset_size': len(df),
                'columns_analyzed': len(df.columns)
            },
            'column_semantics': column_types,
            'metrics': metrics,
            'lagging_areas': {'count': len(lagging), 'items': lagging},
            'booming_areas': {'count': len(booming), 'items': booming},
            'insights': insights
        })
    
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
