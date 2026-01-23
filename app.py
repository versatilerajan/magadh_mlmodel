from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import traceback
import sys

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['JSON_SORT_KEYS'] = False

# Global variables for models
performance_model = None
risk_model = None
scaler = None
label_encoders = None
metadata = {}

def load_models():
    """Load models with error handling"""
    global performance_model, risk_model, scaler, label_encoders, metadata
    
    try:
        print("=" * 60)
        print("Loading analytics models...")
        print("=" * 60)
        
        # Check if models directory exists
        if not os.path.exists('models'):
            print("ERROR: models directory not found!")
            return False
        
        # List files in models directory
        print("\nFiles in models directory:")
        for file in os.listdir('models'):
            print(f"  - {file}")
        
        # Load models
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
        
        print("\n" + "=" * 60)
        print("All models loaded successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR loading models: {e}")
        print(traceback.format_exc())
        return False

# Load models on startup
models_loaded = load_models()

# ==================== HELPER FUNCTIONS ====================

def analyze_column_semantics(df):
    """Analyze column meanings using heuristics and patterns"""
    
    column_types = {}
    
    for col in df.columns:
        col_lower = col.lower()
        sample_values = df[col].dropna().head(100)
        
        # Determine semantic type
        if df[col].dtype in ['int64', 'float64']:
            # Numerical column
            if any(keyword in col_lower for keyword in ['revenue', 'sales', 'income', 'earning']):
                column_types[col] = 'revenue'
            elif any(keyword in col_lower for keyword in ['expense', 'cost', 'spend']):
                column_types[col] = 'expense'
            elif any(keyword in col_lower for keyword in ['profit', 'margin', 'net']):
                column_types[col] = 'profit'
            elif any(keyword in col_lower for keyword in ['employee', 'staff', 'headcount', 'team']):
                column_types[col] = 'employee_count'
            elif any(keyword in col_lower for keyword in ['satisfaction', 'rating', 'score', 'nps']):
                column_types[col] = 'satisfaction'
            elif any(keyword in col_lower for keyword in ['product', 'efficiency', 'output']):
                column_types[col] = 'productivity'
            elif any(keyword in col_lower for keyword in ['growth', 'increase', 'change']):
                column_types[col] = 'growth'
            else:
                column_types[col] = 'numerical_metric'
        else:
            # Categorical column
            if len(sample_values) == 0:
                column_types[col] = 'unknown'
                continue
                
            unique_ratio = len(sample_values.unique()) / len(sample_values) if len(sample_values) > 0 else 0
            
            if any(keyword in col_lower for keyword in ['dept', 'department', 'division', 'unit']):
                column_types[col] = 'department'
            elif any(keyword in col_lower for keyword in ['region', 'location', 'area', 'zone', 'territory']):
                column_types[col] = 'region'
            elif any(keyword in col_lower for keyword in ['category', 'type', 'class']):
                column_types[col] = 'category'
            elif unique_ratio < 0.05:  # Low cardinality
                column_types[col] = 'group'
            else:
                column_types[col] = 'identifier'
    
    return column_types

def calculate_comprehensive_metrics(df, column_types):
    """Calculate comprehensive business metrics from any dataset"""
    
    metrics = {
        'overview': {},
        'numerical_summary': {},
        'categorical_summary': {},
        'correlations': []
    }
    
    # Basic overview
    metrics['overview'] = {
        'total_records': int(len(df)),
        'total_columns': int(len(df.columns)),
        'numerical_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
        'categorical_columns': int(len(df.select_dtypes(include=['object']).columns)),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum())
    }
    
    # Numerical summary
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].notna().sum() == 0:
            continue
            
        metrics['numerical_summary'][col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'sum': float(df[col].sum()),
            'type': column_types.get(col, 'numerical_metric')
        }
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].notna().sum() == 0:
            continue
            
        value_counts = df[col].value_counts()
        if len(value_counts) == 0:
            continue
            
        metrics['categorical_summary'][col] = {
            'unique_values': int(df[col].nunique()),
            'most_common': str(value_counts.index[0]),
            'most_common_count': int(value_counts.iloc[0]),
            'distribution': {str(k): int(v) for k, v in value_counts.head(10).items()},
            'type': column_types.get(col, 'category')
        }
    
    # Calculate correlations for numerical columns
    if len(numerical_cols) > 1:
        try:
            corr_matrix = df[numerical_cols].corr()
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value) and abs(corr_value) > 0.5:
                        metrics['correlations'].append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': float(corr_value),
                            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                        })
        except:
            pass
    
    return metrics

def identify_lagging_areas(df, column_types, metrics):
    """Identify underperforming areas/departments"""
    
    lagging_areas = []
    
    # Find grouping columns
    group_cols = [col for col, type_ in column_types.items() 
                  if type_ in ['department', 'region', 'group', 'category']]
    
    # Find performance metrics
    perf_cols = [col for col, type_ in column_types.items() 
                 if type_ in ['revenue', 'profit', 'productivity', 'satisfaction', 'growth']]
    
    if not group_cols or not perf_cols:
        return lagging_areas
    
    # Analyze each grouping
    for group_col in group_cols[:2]:
        for perf_col in perf_cols[:3]:
            try:
                grouped = df.groupby(group_col)[perf_col].agg(['mean', 'median', 'count'])
                overall_mean = df[perf_col].mean()
                
                if np.isnan(overall_mean) or overall_mean == 0:
                    continue
                
                # Find groups below average
                below_avg = grouped[grouped['mean'] < overall_mean * 0.8]
                
                for group_name, row in below_avg.iterrows():
                    gap = overall_mean - row['mean']
                    gap_pct = (gap / overall_mean) * 100
                    
                    lagging_areas.append({
                        'category': group_col,
                        'name': str(group_name),
                        'metric': perf_col,
                        'current_value': float(row['mean']),
                        'average_value': float(overall_mean),
                        'gap': float(gap),
                        'gap_percentage': float(gap_pct),
                        'sample_size': int(row['count']),
                        'severity': 'critical' if gap_pct > 30 else 'moderate' if gap_pct > 15 else 'minor'
                    })
            except:
                continue
    
    lagging_areas.sort(key=lambda x: x['gap_percentage'], reverse=True)
    return lagging_areas[:10]

def identify_booming_areas(df, column_types, metrics):
    """Identify high-performing areas with growth potential"""
    
    booming_areas = []
    
    group_cols = [col for col, type_ in column_types.items() 
                  if type_ in ['department', 'region', 'group', 'category']]
    
    perf_cols = [col for col, type_ in column_types.items() 
                 if type_ in ['revenue', 'profit', 'productivity', 'satisfaction', 'growth']]
    
    if not group_cols or not perf_cols:
        return booming_areas
    
    for group_col in group_cols[:2]:
        for perf_col in perf_cols[:3]:
            try:
                grouped = df.groupby(group_col)[perf_col].agg(['mean', 'median', 'count', 'std'])
                overall_mean = df[perf_col].mean()
                
                if np.isnan(overall_mean) or overall_mean == 0:
                    continue
                
                above_avg = grouped[grouped['mean'] > overall_mean * 1.2]
                
                for group_name, row in above_avg.iterrows():
                    outperformance = row['mean'] - overall_mean
                    outperformance_pct = (outperformance / overall_mean) * 100
                    cv = (row['std'] / row['mean']) * 100 if row['mean'] != 0 else 100
                    
                    booming_areas.append({
                        'category': group_col,
                        'name': str(group_name),
                        'metric': perf_col,
                        'current_value': float(row['mean']),
                        'average_value': float(overall_mean),
                        'outperformance': float(outperformance),
                        'outperformance_percentage': float(outperformance_pct),
                        'consistency': 'high' if cv < 20 else 'moderate' if cv < 40 else 'low',
                        'sample_size': int(row['count']),
                        'potential': 'excellent' if outperformance_pct > 40 else 'good' if outperformance_pct > 25 else 'fair'
                    })
            except:
                continue
    
    booming_areas.sort(key=lambda x: x['outperformance_percentage'], reverse=True)
    return booming_areas[:10]

def generate_actionable_insights(df, column_types, metrics, lagging, booming):
    """Generate specific, actionable recommendations"""
    
    insights = {
        'critical_actions': [],
        'quick_wins': [],
        'strategic_recommendations': [],
        'growth_opportunities': []
    }
    
    # Critical actions
    for lag in lagging[:3]:
        if lag['severity'] == 'critical':
            insights['critical_actions'].append({
                'priority': 'HIGH',
                'area': f"{lag['category']}: {lag['name']}",
                'issue': f"Performing {lag['gap_percentage']:.1f}% below average in {lag['metric']}",
                'recommendation': f"Immediate intervention required. Consider root cause analysis and resource reallocation.",
                'expected_impact': f"Potential improvement of {lag['gap']:.2f} in {lag['metric']}"
            })
    
    # Quick wins
    for lag in lagging[3:6]:
        if lag['severity'] in ['moderate', 'minor']:
            insights['quick_wins'].append({
                'priority': 'MEDIUM',
                'area': f"{lag['category']}: {lag['name']}",
                'opportunity': f"Close {lag['gap_percentage']:.1f}% gap in {lag['metric']}",
                'recommendation': f"Apply best practices from top performers.",
                'estimated_timeline': '1-3 months'
            })
    
    # Growth opportunities
    for boom in booming[:3]:
        insights['growth_opportunities'].append({
            'priority': 'STRATEGIC',
            'area': f"{boom['category']}: {boom['name']}",
            'strength': f"Outperforming by {boom['outperformance_percentage']:.1f}% in {boom['metric']}",
            'recommendation': f"Scale successful strategies. Consider expansion.",
            'consistency': boom['consistency']
        })
    
    # Strategic recommendations
    if metrics.get('correlations'):
        top_corr = metrics['correlations'][0]
        insights['strategic_recommendations'].append({
            'type': 'Leverage Key Relationship',
            'finding': f"Strong correlation between {top_corr['feature_1']} and {top_corr['feature_2']}",
            'recommendation': f"Focus on improving {top_corr['feature_1']} to impact {top_corr['feature_2']}",
            'confidence': 'High'
        })
    
    return insights

# ==================== API ENDPOINTS ====================

@app.route('/')
def home():
    return jsonify({
        'service': 'Magadh Business Analytics Engine',
        'version': '2.0.0',
        'status': 'online' if models_loaded else 'models not loaded',
        'description': 'AI-powered business analytics and insights generation',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/analyze': 'POST - Comprehensive business analysis',
            '/metrics': 'GET - Model metadata'
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
    if models_loaded:
        return jsonify(metadata)
    else:
        return jsonify({'error': 'Models not loaded'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please check server logs.'}), 500
    
    try:
        start_time = datetime.now()
        
        # Get data
        if 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file)
        elif request.is_json:
            data = request.get_json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                return jsonify({'error': 'JSON data must be a list of objects'}), 400
        else:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate
        if df.empty:
            return jsonify({'error': 'Empty dataset'}), 400
        
        if len(df) < 10:
            return jsonify({'error': 'Minimum 10 records required'}), 400
        
        # Analyze
        print(f"Analyzing {len(df)} rows, {len(df.columns)} columns...")
        
        column_types = analyze_column_semantics(df)
        metrics = calculate_comprehensive_metrics(df, column_types)
        lagging_areas = identify_lagging_areas(df, column_types, metrics)
        booming_areas = identify_booming_areas(df, column_types, metrics)
        insights = generate_actionable_insights(df, column_types, metrics, lagging_areas, booming_areas)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'dataset_size': len(df),
                'columns_analyzed': len(df.columns)
            },
            'column_semantics': column_types,
            'metrics': metrics,
            'lagging_areas': {
                'count': len(lagging_areas),
                'items': lagging_areas
            },
            'booming_areas': {
                'count': len(booming_areas),
                'items': booming_areas
            },
            'insights': insights
        }
        
        print(f"✓ Analysis complete in {processing_time:.2f}s")
        return jsonify(response)
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max 50MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Startup check
if __name__ == '__main__':
    if not models_loaded:
        print("\n" + "="*60)
        print("WARNING: Models failed to load!")
        print("The application will start but /analyze will not work.")
        print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\nStarting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
