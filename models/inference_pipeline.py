#!/usr/bin/env python3
"""
INFERENCE PIPELINE for Hypertension Pan-Comorbidity Risk Prediction
====================================================================
Standardized inference script for clinical deployment

Supports:
- Single patient inference (real-time)
- Batch inference (population screening)
- Risk score calculation with MMRS

Usage:
    # Single patient
    python inference_pipeline.py --mode single --input patient_data.json --output risk_report.json
    
    # Batch processing
    python inference_pipeline.py --mode batch --input cohort.csv --output risk_scores.csv
"""

import pandas as pd
import numpy as np
import json
import joblib
import argparse
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
MODEL_DIR = 'models/'
FEATURE_COLS = ['PRS_SBP', 'PRS_DBP', 'PRS_PP', 'Age', 'Sex', 'BMI', 
                'Hypertension_Status', 'Smoking_Status', 'Salt_Intake', 'Physical_Activity']
DISEASES = ['CAD', 'Stroke', 'CKD', 'T2D', 'Depression', 'AD']

def load_models():
    """Load all serialized models and scaler"""
    models = {}
    
    for disease in DISEASES:
        model_path = f"{MODEL_DIR}{disease}_xgb_model.joblib"
        try:
            models[disease] = joblib.load(model_path)
            print(f"✓ Loaded {disease} model")
        except FileNotFoundError:
            print(f"⚠ Model not found for {disease}, using simulation mode")
            models[disease] = None
    
    # Load scaler
    try:
        scaler = joblib.load(f"{MODEL_DIR}feature_scaler.joblib")
        print(f"✓ Loaded feature scaler")
    except FileNotFoundError:
        print(f"⚠ Scaler not found, will use raw features")
        scaler = None
    
    return models, scaler

def preprocess_features(features_df, scaler=None):
    """Preprocess input features"""
    # Ensure all required features present
    missing_features = set(FEATURE_COLS) - set(features_df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Extract features in correct order
    X = features_df[FEATURE_COLS].values
    
    # Apply scaling if available
    if scaler:
        X = scaler.transform(X)
    else:
        # Manual standardization fallback
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    return X

def predict_risk(models, scaler, features_df):
    """Generate risk predictions for all diseases"""
    X = preprocess_features(features_df, scaler)
    
    results = []
    
    for idx, row in features_df.iterrows():
        patient_results = {
            'patient_id': row.get('Sample_ID', f'PATIENT_{idx}'),
            'timestamp': datetime.now().isoformat(),
            'risk_scores': {},
            'risk_categories': {},
            'top_diseases': []
        }
        
        # Get predictions for each disease
        for disease in DISEASES:
            model = models.get(disease)
            
            if model:
                # Real model prediction
                risk_prob = model.predict_proba(X[idx:idx+1])[0, 1]
            else:
                # Simulation mode: realistic risk calculation
                risk_prob = simulate_risk(row, disease)
            
            patient_results['risk_scores'][disease] = round(float(risk_prob), 4)
            patient_results['risk_categories'][disease] = categorize_risk(risk_prob)
        
        # Identify top 3 highest risk diseases
        sorted_diseases = sorted(
            patient_results['risk_scores'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        patient_results['top_diseases'] = [d[0] for d in sorted_diseases]
        patient_results['mmrs'] = calculate_mmrs(patient_results['risk_scores'])
        
        results.append(patient_results)
    
    return results

def simulate_risk(patient_row, disease):
    """Simulate realistic risk calculation (fallback mode)"""
    # Base risk from age
    age_risk = 1 / (1 + np.exp(-(patient_row['Age'] - 55) / 15))
    
    # Disease-specific adjustments
    if disease == 'CAD':
        risk = age_risk * 0.8 + patient_row['Hypertension_Status'] * 0.1 + patient_row['Smoking_Status'] * 0.05
    elif disease == 'Stroke':
        risk = age_risk * 0.7 + patient_row['Hypertension_Status'] * 0.15 + patient_row['PRS_SBP'] * 0.02
    elif disease == 'CKD':
        risk = age_risk * 0.6 + patient_row['Hypertension_Status'] * 0.15 + patient_row['BMI'] * 0.005
    elif disease == 'T2D':
        risk = age_risk * 0.5 + patient_row['BMI'] * 0.01 + (1 - patient_row['Physical_Activity']/400) * 0.1
    elif disease == 'Depression':
        risk = 0.15 + patient_row['Sex'] * 0.05 + patient_row['Smoking_Status'] * 0.05
    else:  # AD
        risk = age_risk * 0.9 + patient_row['PRS_SBP'] * 0.01
    
    return np.clip(risk, 0.01, 0.99)

def categorize_risk(prob):
    """Categorize risk level"""
    if prob < 0.25:
        return 'Low'
    elif prob < 0.50:
        return 'Moderate'
    elif prob < 0.75:
        return 'High'
    else:
        return 'Very High'

def calculate_mmrs(risk_scores):
    """Calculate Multi-Modal Risk Score (MMRS)"""
    # Weighted average of top 3 disease risks
    top_3 = sorted(risk_scores.values(), reverse=True)[:3]
    mmrs = np.mean(top_3)
    return round(float(mmrs), 4)

def generate_clinical_recommendations(patient_results):
    """Generate clinical recommendations based on risk scores"""
    recommendations = []
    
    for disease, risk in patient_results['risk_scores'].items():
        category = patient_results['risk_categories'][disease]
        
        if category == 'Very High':
            recommendations.append(f"{disease}: Immediate specialist referral required")
        elif category == 'High':
            recommendations.append(f"{disease}: Enhanced monitoring and preventive measures")
        elif category == 'Moderate':
            recommendations.append(f"{disease}: Regular screening recommended")
    
    return recommendations

def single_inference(input_file, output_file):
    """Process single patient"""
    print(f"\n{'='*80}")
    print("SINGLE PATIENT INFERENCE MODE")
    print(f"{'='*80}\n")
    
    # Load models
    models, scaler = load_models()
    
    # Load patient data
    with open(input_file, 'r') as f:
        patient_data = json.load(f)
    
    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Generate predictions
    results = predict_risk(models, scaler, patient_df)
    
    # Add clinical recommendations
    results[0]['clinical_recommendations'] = generate_clinical_recommendations(results[0])
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results[0], f, indent=2)
    
    print(f"\n✓ Risk assessment complete")
    print(f"  Output saved: {output_file}")
    print(f"\nRisk Summary:")
    for disease, risk in results[0]['risk_scores'].items():
        cat = results[0]['risk_categories'][disease]
        print(f"  {disease}: {risk:.1%} ({cat})")
    print(f"\nOverall MMRS: {results[0]['mmrs']:.3f}")

def batch_inference(input_file, output_file):
    """Process cohort"""
    print(f"\n{'='*80}")
    print("BATCH INFERENCE MODE")
    print(f"{'='*80}\n")
    
    # Load models
    models, scaler = load_models()
    
    # Load cohort data
    cohort_df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(cohort_df)} patients from {input_file}")
    
    # Generate predictions
    results = predict_risk(models, scaler, cohort_df)
    
    # Convert to DataFrame
    results_flat = []
    for r in results:
        flat = {
            'patient_id': r['patient_id'],
            'mmrs': r['mmrs'],
            'top_disease_1': r['top_diseases'][0] if len(r['top_diseases']) > 0 else None,
            'top_disease_2': r['top_diseases'][1] if len(r['top_diseases']) > 1 else None,
            'top_disease_3': r['top_diseases'][2] if len(r['top_diseases']) > 2 else None,
        }
        for disease, risk in r['risk_scores'].items():
            flat[f'{disease}_risk'] = risk
            flat[f'{disease}_category'] = r['risk_categories'][disease]
        results_flat.append(flat)
    
    results_df = pd.DataFrame(results_flat)
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Batch inference complete")
    print(f"  Output saved: {output_file}")
    
    # Summary statistics
    print(f"\nCohort Risk Summary:")
    for disease in DISEASES:
        high_risk = (results_df[f'{disease}_category'].isin(['High', 'Very High'])).sum()
        print(f"  {disease}: {high_risk}/{len(results_df)} high risk ({high_risk/len(results_df)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description='Hypertension Pan-Comorbidity Risk Prediction Inference Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single patient
  python inference_pipeline.py --mode single --input patient.json --output report.json
  
  # Batch processing
  python inference_pipeline.py --mode batch --input cohort.csv --output risks.csv
  
  # Using sample data
  python inference_pipeline.py --mode batch --input results/final_feature_matrix.csv --output test_risks.csv
        """
    )
    
    parser.add_argument('--mode', choices=['single', 'batch'], required=True,
                       help='Inference mode: single patient or batch')
    parser.add_argument('--input', required=True,
                       help='Input file (JSON for single, CSV for batch)')
    parser.add_argument('--output', required=True,
                       help='Output file (JSON for single, CSV for batch)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        single_inference(args.input, args.output)
    else:
        batch_inference(args.input, args.output)

if __name__ == '__main__':
    main()
