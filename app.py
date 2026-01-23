from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['JSON_SORT_KEYS'] = False

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
            print(f" - {f}")
       
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

# ==================== ADVANCED ANALYSIS FUNCTIONS ====================
def analyze_column_semantics(df):
    """Advanced column semantic analysis"""
    column_types = {}
   
    for col in df.columns:
        col_lower = col.lower()
        sample_values = df[col].dropna().head(100)
       
        if len(sample_values) == 0:
            column_types[col] = 'empty'
            continue
       
        if df[col].dtype in ['int64', 'float64']:
            # Numerical column semantic detection
            if any(k in col_lower for k in ['revenue', 'sales', 'income', 'earning', 'proceeds', 'turnover']):
                column_types[col] = 'revenue'
            elif any(k in col_lower for k in ['expense', 'cost', 'spend', 'expenditure', 'outlay']):
                column_types[col] = 'expense'
            elif any(k in col_lower for k in ['profit', 'margin', 'net', 'earnings', 'gain']):
                column_types[col] = 'profit'
            elif any(k in col_lower for k in ['employee', 'staff', 'headcount', 'team', 'workforce', 'personnel']):
                column_types[col] = 'employee_count'
            elif any(k in col_lower for k in ['satisfaction', 'rating', 'score', 'nps', 'csat', 'feedback']):
                column_types[col] = 'satisfaction'
            elif any(k in col_lower for k in ['product', 'efficiency', 'output', 'performance', 'throughput']):
                column_types[col] = 'productivity'
            elif any(k in col_lower for k in ['growth', 'increase', 'change', 'rate', 'yoy', 'mom']):
                column_types[col] = 'growth'
            elif any(k in col_lower for k in ['customer', 'client', 'user', 'subscriber']):
                column_types[col] = 'customer_metric'
            elif any(k in col_lower for k in ['conversion', 'retention', 'churn', 'attrition']):
                column_types[col] = 'conversion_metric'
            elif any(k in col_lower for k in ['time', 'duration', 'days', 'hours', 'minutes']):
                column_types[col] = 'time_metric'
            elif any(k in col_lower for k in ['quantity', 'volume', 'count', 'number', 'total']):
                column_types[col] = 'quantity'
            else:
                column_types[col] = 'numerical_metric'
        else:
            # Categorical column semantic detection
            unique_ratio = len(sample_values.unique()) / len(sample_values) if len(sample_values) > 0 else 0
           
            if any(k in col_lower for k in ['dept', 'department', 'division', 'unit', 'team']):
                column_types[col] = 'department'
            elif any(k in col_lower for k in ['region', 'location', 'area', 'zone', 'territory', 'geography']):
                column_types[col] = 'region'
            elif any(k in col_lower for k in ['category', 'type', 'class', 'segment', 'group']):
                column_types[col] = 'category'
            elif any(k in col_lower for k in ['product', 'item', 'sku']):
                column_types[col] = 'product'
            elif any(k in col_lower for k in ['status', 'state', 'stage', 'phase']):
                column_types[col] = 'status'
            elif any(k in col_lower for k in ['name', 'title', 'label']):
                column_types[col] = 'name'
            elif any(k in col_lower for k in ['id', 'code', 'key', 'reference']):
                column_types[col] = 'identifier'
            elif unique_ratio < 0.05:
                column_types[col] = 'group'
            else:
                column_types[col] = 'categorical'
   
    return column_types

def calculate_comprehensive_metrics(df, column_types):
    """Calculate comprehensive business metrics"""
    metrics = {
        'overview': {},
        'numerical_summary': {},
        'categorical_summary': {},
        'data_quality': {},
        'correlations': [],
        'trends': {}
    }
   
    # Overview metrics
    metrics['overview'] = {
        'total_records': int(len(df)),
        'total_columns': int(len(df.columns)),
        'numerical_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
        'categorical_columns': int(len(df.select_dtypes(include=['object']).columns)),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    }
   
    # Data quality metrics
    total_cells = len(df) * len(df.columns)
    metrics['data_quality'] = {
        'completeness_percentage': float((1 - df.isnull().sum().sum() / total_cells) * 100) if total_cells > 0 else 100,
        'uniqueness_percentage': float((1 - df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 100,
        'columns_with_missing_data': int((df.isnull().sum() > 0).sum()),
        'rows_with_missing_data': int(df.isnull().any(axis=1).sum())
    }
   
    # Numerical summary with advanced statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].notna().sum() == 0:
            continue
       
        try:
            q25 = float(df[col].quantile(0.25))
            q75 = float(df[col].quantile(0.75))
            iqr = q75 - q25
           
            metrics['numerical_summary'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'sum': float(df[col].sum()),
                'q25': q25,
                'q75': q75,
                'iqr': iqr,
                'coefficient_of_variation': float((df[col].std() / df[col].mean()) * 100) if df[col].mean() != 0 else 0,
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis()),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float((df[col].isnull().sum() / len(df)) * 100),
                'outliers_count': int(((df[col] < q25 - 1.5 * iqr) | (df[col] > q75 + 1.5 * iqr)).sum()),
                'type': column_types.get(col, 'numerical_metric')
            }
        except:
            continue
   
    # Categorical summary with distribution analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].notna().sum() == 0:
            continue
       
        try:
            value_counts = df[col].value_counts()
            if len(value_counts) == 0:
                continue
           
            metrics['categorical_summary'][col] = {
                'unique_values': int(df[col].nunique()),
                'most_common': str(value_counts.index[0]),
                'most_common_count': int(value_counts.iloc[0]),
                'most_common_percentage': float((value_counts.iloc[0] / len(df)) * 100),
                'least_common': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                'least_common_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                'distribution': {str(k): int(v) for k, v in value_counts.head(10).items()},
                'entropy': float(-sum((value_counts / len(df)) * np.log2(value_counts / len(df)))),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float((df[col].isnull().sum() / len(df)) * 100),
                'type': column_types.get(col, 'categorical')
            }
        except:
            continue
   
    # Correlation analysis
    if len(numerical_cols) > 1:
        try:
            corr_matrix = df[numerical_cols].corr()
           
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value) and abs(corr_value) > 0.3:
                        metrics['correlations'].append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': float(corr_value),
                            'strength': 'very_strong' if abs(corr_value) > 0.9 else 'strong' if abs(corr_value) > 0.7 else 'moderate' if abs(corr_value) > 0.5 else 'weak',
                            'direction': 'positive' if corr_value > 0 else 'negative'
                        })
           
            # Sort by absolute correlation value
            metrics['correlations'].sort(key=lambda x: abs(x['correlation']), reverse=True)
        except:
            pass
   
    # Trend detection (if numeric columns exist)
    try:
        for col in numerical_cols[:5]:  # Top 5 numerical columns
            if df[col].notna().sum() > 10:
                values = df[col].dropna().values
                x = np.arange(len(values))
               
                # Simple linear trend
                z = np.polyfit(x, values, 1)
                trend_direction = 'increasing' if z[0] > 0 else 'decreasing' if z[0] < 0 else 'stable'
               
                metrics['trends'][col] = {
                    'direction': trend_direction,
                    'slope': float(z[0]),
                    'change_percentage': float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                }
    except:
        pass
   
    return metrics

def identify_lagging_areas(df, column_types, metrics):
    """Identify underperforming areas with detailed analysis"""
    lagging_areas = []
   
    # Find grouping columns
    group_cols = [col for col, type_ in column_types.items()
                  if type_ in ['department', 'region', 'group', 'category', 'product', 'status']]
   
    # Find performance metrics
    perf_cols = [col for col, type_ in column_types.items()
                 if type_ in ['revenue', 'profit', 'productivity', 'satisfaction', 'growth',
                             'customer_metric', 'conversion_metric']]
   
    if not group_cols or not perf_cols:
        return lagging_areas
   
    # Analyze each grouping
    for group_col in group_cols[:3]:  # Top 3 grouping columns
        for perf_col in perf_cols[:5]:  # Top 5 performance metrics
            try:
                grouped = df.groupby(group_col)[perf_col].agg(['mean', 'median', 'count', 'std', 'min', 'max'])
                overall_mean = df[perf_col].mean()
                overall_median = df[perf_col].median()
               
                if np.isnan(overall_mean) or overall_mean == 0:
                    continue
               
                # Find groups below threshold (80% of average)
                below_avg = grouped[grouped['mean'] < overall_mean * 0.8]
               
                for group_name, row in below_avg.iterrows():
                    if row['count'] < 3:  # Skip groups with too few samples
                        continue
                   
                    gap = overall_mean - row['mean']
                    gap_pct = (gap / overall_mean) * 100
                   
                    # Calculate severity
                    if gap_pct > 40:
                        severity = 'critical'
                    elif gap_pct > 25:
                        severity = 'high'
                    elif gap_pct > 15:
                        severity = 'moderate'
                    else:
                        severity = 'minor'
                   
                    # Calculate consistency (lower is better)
                    cv = (row['std'] / row['mean']) * 100 if row['mean'] != 0 else 100
                   
                    lagging_areas.append({
                        'category': group_col,
                        'name': str(group_name),
                        'metric': perf_col,
                        'current_value': float(row['mean']),
                        'median_value': float(row['median']),
                        'average_value': float(overall_mean),
                        'overall_median': float(overall_median),
                        'gap': float(gap),
                        'gap_percentage': float(gap_pct),
                        'min_value': float(row['min']),
                        'max_value': float(row['max']),
                        'std_deviation': float(row['std']),
                        'coefficient_of_variation': float(cv),
                        'sample_size': int(row['count']),
                        'severity': severity,
                        'priority': 1 if severity == 'critical' else 2 if severity == 'high' else 3
                    })
            except Exception as e:
                continue
   
    # Sort by gap percentage and priority
    lagging_areas.sort(key=lambda x: (x['priority'], -x['gap_percentage']))
   
    return lagging_areas[:15]  # Top 15 lagging areas

def identify_booming_areas(df, column_types, metrics):
    """Identify high-performing areas with growth potential"""
    booming_areas = []
   
    # Find grouping columns
    group_cols = [col for col, type_ in column_types.items()
                  if type_ in ['department', 'region', 'group', 'category', 'product', 'status']]
   
    # Find performance metrics
    perf_cols = [col for col, type_ in column_types.items()
                 if type_ in ['revenue', 'profit', 'productivity', 'satisfaction', 'growth',
                             'customer_metric', 'conversion_metric']]
   
    if not group_cols or not perf_cols:
        return booming_areas
   
    # Analyze each grouping
    for group_col in group_cols[:3]:
        for perf_col in perf_cols[:5]:
            try:
                grouped = df.groupby(group_col)[perf_col].agg(['mean', 'median', 'count', 'std', 'min', 'max'])
                overall_mean = df[perf_col].mean()
                overall_median = df[perf_col].median()
               
                if np.isnan(overall_mean) or overall_mean == 0:
                    continue
               
                # Find groups significantly above average (120% of average)
                above_avg = grouped[grouped['mean'] > overall_mean * 1.2]
               
                for group_name, row in above_avg.iterrows():
                    if row['count'] < 3:  # Skip groups with too few samples
                        continue
                   
                    outperformance = row['mean'] - overall_mean
                    outperformance_pct = (outperformance / overall_mean) * 100
                   
                    # Check for consistency (lower CV = more consistent)
                    cv = (row['std'] / row['mean']) * 100 if row['mean'] != 0 else 100
                   
                    if cv < 15:
                        consistency = 'very_high'
                    elif cv < 25:
                        consistency = 'high'
                    elif cv < 40:
                        consistency = 'moderate'
                    else:
                        consistency = 'low'
                   
                    # Determine potential
                    if outperformance_pct > 50 and cv < 30:
                        potential = 'exceptional'
                    elif outperformance_pct > 40:
                        potential = 'excellent'
                    elif outperformance_pct > 25:
                        potential = 'good'
                    else:
                        potential = 'fair'
                   
                    booming_areas.append({
                        'category': group_col,
                        'name': str(group_name),
                        'metric': perf_col,
                        'current_value': float(row['mean']),
                        'median_value': float(row['median']),
                        'average_value': float(overall_mean),
                        'overall_median': float(overall_median),
                        'outperformance': float(outperformance),
                        'outperformance_percentage': float(outperformance_pct),
                        'min_value': float(row['min']),
                        'max_value': float(row['max']),
                        'std_deviation': float(row['std']),
                        'coefficient_of_variation': float(cv),
                        'consistency': consistency,
                        'sample_size': int(row['count']),
                        'potential': potential,
                        'scalability_score': float((outperformance_pct / 10) * (1 - cv / 100))  # Higher is better
                    })
            except Exception as e:
                continue
   
    # Sort by outperformance and scalability
    booming_areas.sort(key=lambda x: (-x['outperformance_percentage'], -x['scalability_score']))
   
    return booming_areas[:15]  # Top 15 booming areas

def identify_moderate_performers(df, column_types):
    """Identify areas performing close to average"""
    moderate_areas = []
   
    group_cols = [col for col, type_ in column_types.items()
                  if type_ in ['department', 'region', 'category']]
   
    perf_cols = [col for col, type_ in column_types.items()
                 if type_ in ['revenue', 'profit', 'satisfaction']]
   
    for gc in group_cols[:2]:
        for pc in perf_cols[:3]:
            try:
                grouped = df.groupby(gc)[pc].mean()
                overall_mean = df[pc].mean()
               
                for name, val in grouped.items():
                    # Between 90% and 110% of average
                    if overall_mean * 0.9 <= val <= overall_mean * 1.1:
                        moderate_areas.append({
                            'category': gc,
                            'name': str(name),
                            'metric': pc,
                            'current_value': float(val),
                            'average_value': float(overall_mean),
                            'deviation_percentage': float(((val - overall_mean) / overall_mean) * 100),
                            'status': 'stable'
                        })
            except:
                continue
   
    return moderate_areas[:10]

def generate_actionable_insights(df, column_types, metrics, lagging, booming, moderate):
    """Generate comprehensive, actionable recommendations"""
    insights = {
        'critical_actions': [],
        'high_priority_actions': [],
        'quick_wins': [],
        'strategic_recommendations': [],
        'growth_opportunities': [],
        'optimization_suggestions': [],
        'risk_alerts': []
    }
   
    # Critical actions for severe underperformers
    for lag in lagging:
        if lag['severity'] == 'critical':
            recovery_potential = lag['gap']
           
            insights['critical_actions'].append({
                'priority': 'CRITICAL',
                'urgency': 'IMMEDIATE',
                'area': f"{lag['category']}: {lag['name']}",
                'issue': f"Performing {lag['gap_percentage']:.1f}% below average in {lag['metric']}",
                'current_performance': f"{lag['current_value']:.2f}",
                'target_performance': f"{lag['average_value']:.2f}",
                'gap_to_close': f"{lag['gap']:.2f}",
                'recommendation': "IMMEDIATE INTERVENTION REQUIRED:\n• Conduct root cause analysis within 48 hours\n• Deploy task force for recovery plan\n• Consider emergency resource allocation\n• Set weekly performance checkpoints",
                'expected_impact': f"Recovery potential: {recovery_potential:.2f} in {lag['metric']}",
                'timeline': '0-2 weeks',
                'resources_needed': 'High - Senior management attention required'
            })
       
        elif lag['severity'] == 'high':
            insights['high_priority_actions'].append({
                'priority': 'HIGH',
                'urgency': 'URGENT',
                'area': f"{lag['category']}: {lag['name']}",
                'issue': f"{lag['gap_percentage']:.1f}% below average in {lag['metric']}",
                'recommendation': "Priority improvement needed:\n• Identify performance gaps\n• Implement best practices from top performers\n• Increase monitoring frequency\n• Allocate additional support",
                'estimated_timeline': '2-4 weeks'
            })
   
    # Quick wins from moderate underperformers
    for lag in lagging:
        if lag['severity'] in ['moderate', 'minor'] and lag['sample_size'] >= 5:
            insights['quick_wins'].append({
                'priority': 'MEDIUM',
                'area': f"{lag['category']}: {lag['name']}",
                'opportunity': f"Close {lag['gap_percentage']:.1f}% performance gap",
                'current_vs_target': f"{lag['current_value']:.2f} → {lag['average_value']:.2f}",
                'recommendation': "Quick improvement opportunities:\n• Adopt proven practices from top performers\n• Provide targeted training\n• Optimize resource allocation\n• Share success stories internally",
                'estimated_effort': 'Low to Medium',
                'estimated_timeline': '1-2 months',
                'success_probability': 'High'
            })
   
    # Growth opportunities from top performers
    for boom in booming:
        if boom['potential'] in ['exceptional', 'excellent']:
            insights['growth_opportunities'].append({
                'priority': 'STRATEGIC',
                'area': f"{boom['category']}: {boom['name']}",
                'strength': f"Outperforming by {boom['outperformance_percentage']:.1f}% in {boom['metric']}",
                'performance_level': f"{boom['current_value']:.2f} vs average {boom['average_value']:.2f}",
                'consistency': boom['consistency'].replace('_', ' ').title(),
                'recommendation': f"SCALE SUCCESS STRATEGY:\n• Document and standardize successful practices\n• Replicate model to similar areas\n• Increase investment by 20-30%\n• Use as benchmark for training\n• Consider expansion opportunities",
                'potential_multiplier': f"{(boom['outperformance_percentage']/100 + 1):.2f}x",
                'scalability_score': f"{boom['scalability_score']:.2f}/10",
                'next_steps': '• Study success factors\n• Create replication playbook\n• Pilot in 2-3 similar areas'
            })
   
    # Optimization for moderate performers
    for mod in moderate[:5]:
        insights['optimization_suggestions'].append({
            'priority': 'LOW',
            'area': f"{mod['category']}: {mod['name']}",
            'status': 'Stable performer - optimization opportunity',
            'current_performance': f"{mod['current_value']:.2f}",
            'recommendation': "Incremental improvement:\n• Fine-tune existing processes\n• Implement small optimizations\n• Monitor for drift\n• Maintain current standards",
            'expected_improvement': '5-10%'
        })
   
    # Strategic recommendations based on correlations
    if metrics.get('correlations'):
        top_correlations = [c for c in metrics['correlations'] if c['strength'] in ['strong', 'very_strong']][:3]
       
        for i, corr in enumerate(top_correlations, 1):
            insights['strategic_recommendations'].append({
                'type': 'Leverage Key Relationship',
                'rank': i,
                'finding': f"{'Very strong' if corr['strength'] == 'very_strong' else 'Strong'} {corr['direction']} correlation ({corr['correlation']:.3f}) between {corr['feature_1']} and {corr['feature_2']}",
                'implication': f"Changes in {corr['feature_1']} will significantly impact {corr['feature_2']}",
                'recommendation': f"{'Increase' if corr['direction'] == 'positive' else 'Optimize'} {corr['feature_1']} to {'improve' if corr['direction'] == 'positive' else 'balance'} {corr['feature_2']}",
                'confidence': 'Very High' if abs(corr['correlation']) > 0.8 else 'High',
                'action_items': [
                    f"Focus improvement efforts on {corr['feature_1']}",
                    f"Monitor {corr['feature_2']} as leading indicator",
                    f"Create dashboard tracking both metrics together"
                ]
            })
   
    # Risk alerts based on data quality and outliers
    if metrics['data_quality']['completeness_percentage'] < 95:
        insights['risk_alerts'].append({
            'type': 'Data Quality Risk',
            'severity': 'HIGH' if metrics['data_quality']['completeness_percentage'] < 90 else 'MEDIUM',
            'issue': f"Data completeness is only {metrics['data_quality']['completeness_percentage']:.1f}%",
            'impact': 'Analysis reliability may be compromised',
            'recommendation': 'Improve data collection processes and fill missing values'
        })
   
    # Overall strategic direction
    total_lagging = len([l for l in lagging if l['severity'] in ['critical', 'high']])
    total_booming = len([b for b in booming if b['potential'] in ['exceptional', 'excellent']])
   
    if total_lagging > total_booming:
        insights['strategic_recommendations'].insert(0, {
            'type': 'Organizational Focus',
            'priority': 'HIGH',
            'finding': f"{total_lagging} critical/high-priority underperforming areas vs {total_booming} exceptional performers",
            'recommendation': "TURNAROUND STRATEGY NEEDED:\n• Form dedicated improvement task force\n• Reallocate resources from stable to struggling areas\n• Implement 90-day rapid improvement plan\n• Weekly executive review meetings",
            'expected_outcome': 'Stabilize underperformers within 3 months'
        })
    elif total_booming > total_lagging * 2:
        insights['strategic_recommendations'].insert(0, {
            'type': 'Organizational Focus',
            'priority': 'STRATEGIC',
            'finding': f"{total_booming} high-performing areas identified - strong growth position",
            'recommendation': "GROWTH ACCELERATION STRATEGY:\n• Scale successful models organization-wide\n• Increase investment in top performers\n• Document and replicate best practices\n• Position for market expansion",
            'expected_outcome': 'Accelerate growth by 25-40% in next quarter'
        })
   
    return insights

def generate_executive_summary(df, metrics, lagging, booming, insights):
    """Generate executive-level summary"""
    summary = {
        'overall_health_score': 0,
        'key_findings': [],
        'top_3_priorities': [],
        'performance_distribution': {},
        'recommended_focus_areas': []
    }
   
    # Calculate overall health score (0-100)
    critical_count = len([l for l in lagging if l['severity'] == 'critical'])
    high_count = len([l for l in lagging if l['severity'] == 'high'])
    excellent_count = len([b for b in booming if b['potential'] in ['exceptional', 'excellent']])
   
    # Score calculation
    health_score = 70  # Base score
    health_score -= (critical_count * 10)  # -10 for each critical issue
    health_score -= (high_count * 5)  # -5 for each high priority issue
    health_score += (excellent_count * 3)  # +3 for each excellent performer
    health_score = max(0, min(100, health_score))  # Clamp between 0-100

    summary['overall_health_score'] = float(health_score)

    # Health status
    if health_score >= 80:
        summary['health_status'] = 'Excellent'
        summary['health_description'] = 'Organization is performing well with strong growth potential'
    elif health_score >= 60:
        summary['health_status'] = 'Good'
        summary['health_description'] = 'Solid performance with some areas needing attention'
    elif health_score >= 40:
        summary['health_status'] = 'Fair'
        summary['health_description'] = 'Multiple areas require improvement'
    else:
        summary['health_status'] = 'Poor'
        summary['health_description'] = 'Urgent intervention needed across multiple areas'

    # Key findings
    summary['key_findings'] = [
        f"Analyzed {metrics['overview']['total_records']:,} records across {metrics['overview']['total_columns']} dimensions",
        f"Identified {len(lagging)} underperforming areas and {len(booming)} high performers",
        f"Data quality: {metrics['data_quality']['completeness_percentage']:.1f}% complete",
        f"Found {len(metrics.get('correlations', []))} significant correlations between metrics"
    ]

    # Top 3 priorities
    all_actions = (
        [(a, 'CRITICAL') for a in insights.get('critical_actions', [])] +
        [(a, 'HIGH') for a in insights.get('high_priority_actions', [])] +
        [(a, 'GROWTH') for a in insights.get('growth_opportunities', [])]
    )

    for action, action_type in all_actions[:3]:
        summary['top_3_priorities'].append({
            'type': action_type,
            'area': action.get('area', 'N/A'),
            'action': action.get('recommendation', '').split('\n')[0]  # First line only
        })

    return summary

# ==================== API ENDPOINTS ====================
@app.route('/')
def home():
    return jsonify({
        'service': 'Magadh Business Analytics Engine',
        'version': '2.0.0',
        'status': 'online' if models_loaded else 'models not loaded',
        'description': 'AI-powered comprehensive business analytics and insights generation',
        'capabilities': [
            'Dynamic CSV analysis (any column structure)',
            'Advanced semantic column detection',
            'Performance prediction & risk assessment',
            'Lagging area identification with severity levels',
            'High-performer & growth opportunity analysis',
            'Moderate performer tracking',
            'Actionable insights with priority levels',
            'Correlation & trend analysis',
            'Data quality assessment',
            'Executive summary generation'
        ],
        'endpoints': {
            '/': 'API information',
            '/health': 'GET - Health check & status',
            '/metrics': 'GET - Model metadata & performance',
            '/analyze': 'POST - Comprehensive business analysis (File or JSON)',
            '/analyze/summary': 'POST - Quick executive summary',
            '/analyze/detailed': 'POST - Full detailed analysis report'
        },
        'supported_formats': ['CSV file upload', 'JSON array'],
        'max_file_size': '100MB',
        'average_processing_time': '5-30 seconds depending on dataset size'
    })

@app.route('/health')
def health():
    health_status = {
        'status': 'healthy' if models_loaded else 'degraded',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat(),
        'uptime_status': 'operational',
        'api_version': '2.0.0'
    }

    if models_loaded:
        health_status['model_details'] = {
            'performance_model': 'loaded',
            'risk_model': 'loaded',
            'scaler': 'loaded',
            'encoders': 'loaded'
        }

    return jsonify(health_status)

@app.route('/metrics')
def get_metrics():
    if models_loaded:
        return jsonify({
            'model_metadata': metadata,
            'service_info': {
                'version': '2.0.0',
                'last_updated': datetime.now().isoformat(),
                'analysis_capabilities': [
                    'semantic_analysis',
                    'performance_metrics',
                    'trend_detection',
                    'correlation_analysis',
                    'anomaly_detection'
                ]
            }
        })
    else:
        return jsonify({'error': 'Models not loaded', 'status': 'unavailable'}), 500

@app.route('/analyze', methods=['POST'])
@app.route('/analyze/detailed', methods=['POST'])
def analyze_detailed():
    """Main comprehensive analysis endpoint"""

    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded',
            'message': 'The analytics engine is currently unavailable. Please contact support.'
        }), 500

    try:
        start_time = datetime.now()
        
        # Get data from request
        if 'file' in request.files:
            file = request.files['file']
            print(f"Analyzing uploaded file: {file.filename}")
            df = pd.read_csv(file)
        elif request.is_json:
            data = request.get_json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                return jsonify({'error': 'JSON data must be an array of objects'}), 400
        else:
            return jsonify({'error': 'No data provided. Send CSV file or JSON array'}), 400
        
        # Validate dataset
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        if len(df) < 5:
            return jsonify({'error': 'Dataset too small. Minimum 5 records required'}), 400
        
        if len(df) > 1000000:
            return jsonify({'error': 'Dataset too large. Maximum 1,000,000 records allowed'}), 400
        
        print(f"Processing dataset: {len(df)} rows × {len(df.columns)} columns")
        
        # Step 1: Analyze column semantics
        print("Step 1/7: Analyzing column semantics...")
        column_types = analyze_column_semantics(df)
        
        # Step 2: Calculate comprehensive metrics
        print("Step 2/7: Calculating metrics...")
        metrics = calculate_comprehensive_metrics(df, column_types)
        
        # Step 3: Identify lagging areas
        print("Step 3/7: Identifying lagging areas...")
        lagging_areas = identify_lagging_areas(df, column_types, metrics)
        
        # Step 4: Identify booming areas
        print("Step 4/7: Identifying booming areas...")
        booming_areas = identify_booming_areas(df, column_types, metrics)
        
        # Step 5: Identify moderate performers
        print("Step 5/7: Analyzing moderate performers...")
        moderate_areas = identify_moderate_performers(df, column_types)
        
        # Step 6: Generate actionable insights
        print("Step 6/7: Generating actionable insights...")
        insights = generate_actionable_insights(df, column_types, metrics, lagging_areas, booming_areas, moderate_areas)
        
        # Step 7: Generate executive summary
        print("Step 7/7: Creating executive summary...")
        executive_summary = generate_executive_summary(df, metrics, lagging_areas, booming_areas, insights)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Compile comprehensive response
        response = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'dataset_info': {
                    'total_records': len(df),
                    'total_columns': len(df.columns),
                    'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                    'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                },
                'analysis_version': '2.0.0',
                'analysis_type': 'comprehensive'
            },
            
            'executive_summary': executive_summary,
            
            'column_semantics': column_types,
            
            'metrics': metrics,
            
            'performance_analysis': {
                'lagging_areas': {
                    'total_count': len(lagging_areas),
                    'critical_count': len([l for l in lagging_areas if l['severity'] == 'critical']),
                    'high_priority_count': len([l for l in lagging_areas if l['severity'] == 'high']),
                    'moderate_count': len([l for l in lagging_areas if l['severity'] == 'moderate']),
                    'items': lagging_areas
                },
                'booming_areas': {
                    'total_count': len(booming_areas),
                    'exceptional_count': len([b for b in booming_areas if b['potential'] == 'exceptional']),
                    'excellent_count': len([b for b in booming_areas if b['potential'] == 'excellent']),
                    'items': booming_areas
                },
                'moderate_performers': {
                    'total_count': len(moderate_areas),
                    'items': moderate_areas
                }
            },
            
            'actionable_insights': insights,
            
            'insights_summary': {
                'total_recommendations': sum([
                    len(insights.get('critical_actions', [])),
                    len(insights.get('high_priority_actions', [])),
                    len(insights.get('quick_wins', [])),
                    len(insights.get('strategic_recommendations', [])),
                    len(insights.get('growth_opportunities', [])),
                    len(insights.get('optimization_suggestions', []))
                ]),
                'by_priority': {
                    'critical': len(insights.get('critical_actions', [])),
                    'high': len(insights.get('high_priority_actions', [])),
                    'medium': len(insights.get('quick_wins', [])),
                    'strategic': len(insights.get('strategic_recommendations', [])),
                    'growth': len(insights.get('growth_opportunities', [])),
                    'optimization': len(insights.get('optimization_suggestions', []))
                },
                'risk_alerts': len(insights.get('risk_alerts', []))
            }
        }
        
        print(f"✓ Analysis complete in {processing_time:.2f} seconds")
        print(f"  - Identified {len(lagging_areas)} lagging areas")
        print(f"  - Identified {len(booming_areas)} booming areas")
        print(f"  - Generated {response['insights_summary']['total_recommendations']} recommendations")
        
        return jsonify(response)

    except pd.errors.ParserError:
        return jsonify({
            'error': 'Invalid CSV format',
            'message': 'Unable to parse the CSV file. Please check the file format.'
        }), 400

    except Exception as e:
        print(f"✗ Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'type': type(e).__name__,
            'details': 'Please check your data format and try again'
        }), 500

@app.route('/analyze/summary', methods=['POST'])
def analyze_summary():
    """Quick executive summary endpoint"""

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
        
        # Quick analysis
        column_types = analyze_column_semantics(df)
        metrics = calculate_comprehensive_metrics(df, column_types)
        lagging = identify_lagging_areas(df, column_types, metrics)[:5]
        booming = identify_booming_areas(df, column_types, metrics)[:5]
        insights = generate_actionable_insights(df, column_types, metrics, lagging, booming, [])
        summary = generate_executive_summary(df, metrics, lagging, booming, insights)
        
        return jsonify({
            'executive_summary': summary,
            'quick_stats': {
                'records_analyzed': len(df),
                'lagging_areas_found': len(lagging),
                'booming_areas_found': len(booming),
                'critical_actions_needed': len(insights.get('critical_actions', []))
            },
            'top_priorities': summary.get('top_3_priorities', [])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': 'Maximum file size is 100MB',
        'suggestion': 'Please reduce the dataset size or contact support for enterprise options'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'suggestion': 'Please try again or contact support if the issue persists'
    }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': ['/health', '/metrics', '/analyze', '/analyze/summary', '/analyze/detailed']
    }), 404

# Startup message
if __name__ == '__main__':
    if not models_loaded:
        print("\n" + "="*60)
        print("WARNING: Models failed to load!")
        print("Analysis endpoints will not work properly.")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("Magadh Business Analytics Engine Ready!")
        print("All systems operational")
        print("="*60 + "\n")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
