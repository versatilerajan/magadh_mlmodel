from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import traceback

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['JSON_SORT_KEYS'] = False

# Load models
print("Loading analytics models...")
try:
    performance_model = joblib.load('models/performance_model.pkl')
    risk_model = joblib.load('models/risk_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    performance_model = None
    risk_model = None
    scaler = None
    metadata = {}

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
            unique_ratio = len(sample_values.unique()) / len(sample_values)
            
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
        'trends': {},
        'correlations': []
    }
    
    # Basic overview
    metrics['overview'] = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Numerical summary
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
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
        value_counts = df[col].value_counts()
        metrics['categorical_summary'][col] = {
            'unique_values': int(df[col].nunique()),
            'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'distribution': value_counts.head(10).to_dict(),
            'type': column_types.get(col, 'category')
        }
    
    # Calculate correlations for numerical columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        
        # Find strong correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Strong correlation
                    metrics['correlations'].append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })
    
    return metrics

def identify_lagging_areas(df, column_types, metrics):
    """Identify underperforming areas/departments"""
    
    lagging_areas = []
    
    # Find grouping columns (department, region, etc.)
    group_cols = [col for col, type_ in column_types.items() 
                  if type_ in ['department', 'region', 'group', 'category']]
    
    # Find performance metrics
    perf_cols = [col for col, type_ in column_types.items() 
                 if type_ in ['revenue', 'profit', 'productivity', 'satisfaction', 'growth']]
    
    if not group_cols or not perf_cols:
        return lagging_areas
    
    # Analyze each grouping
    for group_col in group_cols[:2]:  # Limit to first 2 grouping columns
        for perf_col in perf_cols[:3]:  # Limit to first 3 performance metrics
            try:
                grouped = df.groupby(group_col)[perf_col].agg(['mean', 'median', 'count'])
                overall_mean = df[perf_col].mean()
                
                # Find groups below average
                below_avg = grouped[grouped['mean'] < overall_mean * 0.8]  # 20% below average
                
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
            except Exception as e:
                continue
    
    # Sort by gap percentage
    lagging_areas.sort(key=lambda x: x['gap_percentage'], reverse=True)
    
    return lagging_areas[:10]  # Return top 10

def identify_booming_areas(df, column_types, metrics):
    """Identify high-performing areas with growth potential"""
    
    booming_areas = []
    
    # Find grouping columns
    group_cols = [col for col, type_ in column_types.items() 
                  if type_ in ['department', 'region', 'group', 'category']]
    
    # Find performance metrics
    perf_cols = [col for col, type_ in column_types.items() 
                 if type_ in ['revenue', 'profit', 'productivity', 'satisfaction', 'growth']]
    
    if not group_cols or not perf_cols:
        return booming_areas
    
    # Analyze each grouping
    for group_col in group_cols[:2]:
        for perf_col in perf_cols[:3]:
            try:
                grouped = df.groupby(group_col)[perf_col].agg(['mean', 'median', 'count', 'std'])
                overall_mean = df[perf_col].mean()
                
                # Find groups significantly above average
                above_avg = grouped[grouped['mean'] > overall_mean * 1.2]  # 20% above average
                
                for group_name, row in above_avg.iterrows():
                    outperformance = row['mean'] - overall_mean
                    outperformance_pct = (outperformance / overall_mean) * 100
                    
                    # Check for consistency (lower std relative to mean)
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
                        'coefficient_of_variation': float(cv),
                        'sample_size': int(row['count']),
                        'potential': 'excellent' if outperformance_pct > 40 else 'good' if outperformance_pct > 25 else 'fair'
                    })
            except Exception as e:
                continue
    
    # Sort by outperformance
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
    
    # Critical actions for lagging areas
    for lag in lagging[:3]:  # Top 3 lagging areas
        if lag['severity'] == 'critical':
            insights['critical_actions'].append({
                'priority': 'HIGH',
                'area': f"{lag['category']}: {lag['name']}",
                'issue': f"Performing {lag['gap_percentage']:.1f}% below average in {lag['metric']}",
                'recommendation': f"Immediate intervention required. Consider: root cause analysis, resource reallocation, or process improvement initiatives.",
                'expected_impact': f"Potential to recover ${lag['gap']:,.0f} in {lag['metric']}" if 'revenue' in lag['metric'].lower() or 'profit' in lag['metric'].lower() else f"Improve {lag['metric']} by {lag['gap']:.1f} points"
            })
    
    # Quick wins from moderate performers
    for lag in lagging[3:6]:
        if lag['severity'] in ['moderate', 'minor']:
            insights['quick_wins'].append({
                'priority': 'MEDIUM',
                'area': f"{lag['category']}: {lag['name']}",
                'opportunity': f"Close {lag['gap_percentage']:.1f}% gap in {lag['metric']}",
                'recommendation': f"Apply best practices from top performers. Quick process improvements could yield significant results.",
                'estimated_effort': 'Low to Medium',
                'estimated_timeline': '1-3 months'
            })
    
    # Growth opportunities from booming areas
    for boom in booming[:3]:
        insights['growth_opportunities'].append({
            'priority': 'STRATEGIC',
            'area': f"{boom['category']}: {boom['name']}",
            'strength': f"Outperforming by {boom['outperformance_percentage']:.1f}% in {boom['metric']}",
            'recommendation': f"Scale successful strategies. Consider: increased investment, expansion, or replicating model to other areas.",
            'potential_multiplier': f"{boom['outperformance_percentage']/100 + 1:.1f}x",
            'consistency': boom['consistency']
        })
    
    # Strategic recommendations based on correlations
    if metrics.get('correlations'):
        strong_corrs = [c for c in metrics['correlations'] if c['strength'] == 'strong']
        if strong_corrs:
            top_corr = strong_corrs[0]
            insights['strategic_recommendations'].append({
                'type': 'Leverage Key Relationship',
                'finding': f"Strong correlation ({top_corr['correlation']:.2f}) between {top_corr['feature_1']} and {top_corr['feature_2']}",
                'recommendation': f"Focus on improving {top_corr['feature_1']} to positively impact {top_corr['feature_2']}",
                'confidence': 'High'
            })
    
    # Overall strategic direction
    if lagging:
        total_recovery_potential = sum(lag.get('gap', 0) for lag in lagging if 'revenue' in lag.get('metric', '').lower())
        if total_recovery_potential > 0:
            insights['strategic_recommendations'].append({
                'type': 'Revenue Recovery',
                'finding': f"Total revenue recovery potential: ${total_recovery_potential:,.0f}",
                'recommendation': "Implement turnaround strategy for underperforming areas with dedicated task force",
                'priority': 'HIGH'
            })
    
    return insights

# ==================== API ENDPOINTS ====================

@app.route('/')
def home():
    return jsonify({
        'service': 'Magadh Business Analytics Engine',
        'version': '2.0.0',
        'description': 'AI-powered business analytics and insights generation',
        'capabilities': [
            'Dynamic CSV analysis (any columns)',
            'Performance prediction',
            'Lagging area identification',
            'Booming sector analysis',
            'Actionable insights generation',
            'Correlation analysis',
            'Risk categorization'
        ],
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
        'status': 'healthy',
        'models_loaded': performance_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/metrics')
def get_metrics():
    return jsonify(metadata)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint - accepts CSV data and returns comprehensive insights"""
    
    try:
        start_time = datetime.now()
        
        # Get data from request
        if 'file' in request.files:
            # File upload
            file = request.files['file']
            df = pd.read_csv(file)
        elif request.is_json:
            # JSON data
            data = request.get_json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                return jsonify({'error': 'JSON data must be a list of objects'}), 400
        else:
            return jsonify({'error': 'No data provided. Send CSV file or JSON data'}), 400
        
        # Validate data
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        if len(df) < 10:
            return jsonify({'error': 'Dataset too small. Minimum 10 records required'}), 400
        
        # Step 1: Analyze column semantics
        print(f"Analyzing dataset with {len(df)} rows and {len(df.columns)} columns...")
        column_types = analyze_column_semantics(df)
        
        # Step 2: Calculate comprehensive metrics
        print("Calculating metrics...")
        metrics = calculate_comprehensive_metrics(df, column_types)
        
        # Step 3: Identify lagging areas
        print("Identifying lagging areas...")
        lagging_areas = identify_lagging_areas(df, column_types, metrics)
        
        # Step 4: Identify booming areas
        print("Identifying booming areas...")
        booming_areas = identify_booming_areas(df, column_types, metrics)
        
        # Step 5: Generate actionable insights
        print("Generating insights...")
        insights = generate_actionable_insights(df, column_types, metrics, lagging_areas, booming_areas)
        
        # Step 6: Statistical analysis
        statistical_summary = {
            'data_quality': {
                'completeness': float((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                'uniqueness': float((1 - df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 100,
                'total_records': len(df),
                'usable_records': len(df.dropna())
            },
            'distribution_analysis': {}
        }
        
        # Analyze distribution of numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols[:5]:  # Top 5 numerical columns
            skewness = float(df[col].skew())
            statistical_summary['distribution_analysis'][col] = {
                'skewness': skewness,
                'distribution_type': 'right-skewed' if skewness > 0.5 else 'left-skewed' if skewness < -0.5 else 'normal',
                'outliers_count': int(((df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) |
                                       (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))).sum())
            }
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Compile comprehensive response
        response = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'dataset_size': len(df),
                'columns_analyzed': len(df.columns),
                'analysis_version': '2.0.0'
            },
            'column_semantics': column_types,
            'metrics': metrics,
            'statistical_summary': statistical_summary,
            'lagging_areas': {
                'count': len(lagging_areas),
                'items': lagging_areas,
                'summary': f"Found {len(lagging_areas)} underperforming areas requiring attention"
            },
            'booming_areas': {
                'count': len(booming_areas),
                'items': booming_areas,
                'summary': f"Identified {len(booming_areas)} high-performing areas with growth potential"
            },
            'insights': insights,
            'recommendations_summary': {
                'critical_actions': len(insights['critical_actions']),
                'quick_wins': len(insights['quick_wins']),
                'strategic_recommendations': len(insights['strategic_recommendations']),
                'growth_opportunities': len(insights['growth_opportunities']),
                'total_recommendations': sum([
                    len(insights['critical_actions']),
                    len(insights['quick_wins']),
                    len(insights['strategic_recommendations']),
                    len(insights['growth_opportunities'])
                ])
            }
        }
        
        print(f"✓ Analysis complete in {processing_time:.2f} seconds")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"✗ Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'type': type(e).__name__
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
