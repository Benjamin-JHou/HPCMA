"""
FastAPI Inference Server for Hypertension Comorbidity Risk Prediction
Production-ready REST API for real-time risk scoring
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np
import json
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MMRP Clinical AI API",
    description="Multi-Modal Risk Prediction for Hypertension Comorbidities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
# SECURITY NOTE: In production, replace ["*"] with specific allowed origins
# e.g., allow_origins=["https://your-institution.edu", "https://app.yoursite.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict to needed methods only
    allow_headers=["Content-Type", "Authorization"],  # Restrict to needed headers
)

# Request/Response Models
class PatientFeatures(BaseModel):
    """Input features for risk prediction"""
    sbp_prs: float = Field(..., ge=-5, le=5, description="SBP Polygenic Risk Score (z-score)")
    dbp_prs: float = Field(..., ge=-5, le=5, description="DBP Polygenic Risk Score (z-score)")
    pp_prs: float = Field(..., ge=-5, le=5, description="Pulse Pressure PRS (z-score)")
    age: float = Field(..., ge=30, le=85, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    bmi: float = Field(..., ge=15, le=50, description="Body Mass Index (kg/m²)")
    sbp: float = Field(..., ge=90, le=200, description="Systolic Blood Pressure (mmHg)")
    smoking: int = Field(..., ge=0, le=1, description="Smoking status (0=No, 1=Yes)")
    physical_activity: float = Field(..., ge=0, le=10, description="Physical activity score (0-10)")
    diet_score: float = Field(..., ge=0, le=10, description="Diet quality score (0-10)")

class RiskPrediction(BaseModel):
    """Risk prediction output"""
    disease: str
    probability: float = Field(..., ge=0, le=1)
    risk_category: str
    confidence_interval: tuple
    
class PredictionResponse(BaseModel):
    """Full prediction response"""
    patient_id: Optional[str]
    timestamp: str
    individual_risks: Dict[str, RiskPrediction]
    mmrs_score: float
    mmrs_category: str
    top_predictors: Dict[str, List[Dict]]
    clinical_actions: Dict[str, List[str]]
    
class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientFeatures]
    patient_ids: Optional[List[str]] = None

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    models_loaded: bool

# Load model configuration
CONFIG_PATH = os.environ.get('CONFIG_PATH', '/app/config')
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models')

def load_config():
    """Load model configuration"""
    try:
        with open(f"{MODEL_PATH}/model_version_metadata.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load model config: {e}")
        return {}

def load_risk_interpretation():
    """Load risk interpretation table"""
    try:
        with open(f"{MODEL_PATH}/risk_score_interpretation_table.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load risk interpretation: {e}")
        return {}

def load_clinical_mapping():
    """Load clinical action mapping"""
    try:
        with open(f"{MODEL_PATH}/clinical_action_mapping.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load clinical mapping: {e}")
        return {}

# Initialize configurations
model_config = load_config()
risk_interpretation = load_risk_interpretation()
clinical_mapping = load_clinical_mapping()

# Simulate model prediction (replace with actual XGBoost models)
def predict_disease_risk(features: PatientFeatures, disease: str) -> RiskPrediction:
    """
    Predict risk for a specific disease
    NOTE: This is a simulation. Replace with actual XGBoost model inference.
    """
    # Extract features
    f = features
    
    # Simulate prediction based on feature importance
    # This would be replaced with actual model.predict() call
    base_risk = 0.3
    
    # Age factor (increases risk)
    age_factor = (f.age - 50) / 100
    
    # BP factor
    bp_factor = (f.sbp - 120) / 200
    
    # PRS factor (genetic predisposition)
    prs_factor = (f.sbp_prs + f.dbp_prs + f.pp_prs) / 10
    
    # BMI factor
    bmi_factor = (f.bmi - 25) / 50
    
    # Smoking factor
    smoke_factor = f.smoking * 0.15
    
    # Disease-specific adjustments
    disease_multipliers = {
        'CAD': 1.0,
        'Stroke': 0.9,
        'CKD': 1.1,
        'T2D': 0.95,
        'Depression': 0.85,
        'AD': 0.8
    }
    
    multiplier = disease_multipliers.get(disease, 1.0)
    
    # Calculate probability
    probability = base_risk + (
        age_factor * 0.2 +
        bp_factor * 0.25 +
        prs_factor * 0.2 +
        bmi_factor * 0.15 +
        smoke_factor * 0.1
    ) * multiplier
    
    # Clip to valid range
    probability = max(0.0, min(1.0, probability))
    
    # Determine risk category
    if probability < 0.15:
        category = "Low"
    elif probability < 0.3:
        category = "Moderate"
    elif probability < 0.45:
        category = "High"
    else:
        category = "Very High"
    
    # Calculate confidence interval (simulated)
    ci_width = 0.05 + (1 - probability) * 0.05
    ci_lower = max(0, probability - ci_width)
    ci_upper = min(1, probability + ci_width)
    
    return RiskPrediction(
        disease=disease,
        probability=round(probability, 4),
        risk_category=category,
        confidence_interval=(round(ci_lower, 4), round(ci_upper, 4))
    )

def calculate_mmrs(individual_risks: Dict[str, RiskPrediction]) -> tuple:
    """Calculate Multi-Modal Risk Score"""
    probabilities = [r.probability for r in individual_risks.values()]
    
    # Weighted average (equal weights for now)
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    weighted_sum = sum(p * w for p, w in zip(probabilities, weights))
    total_weight = sum(weights)
    
    mmrs = weighted_sum / total_weight
    
    # Determine category
    if mmrs < 0.2:
        category = "Low Risk"
    elif mmrs < 0.35:
        category = "Moderate Risk"
    elif mmrs < 0.5:
        category = "High Risk"
    else:
        category = "Very High Risk"
    
    return round(mmrs, 4), category

def get_top_predictors(features: PatientFeatures) -> Dict[str, List[Dict]]:
    """Get top predictive features for each disease"""
    # This would use actual SHAP values from trained models
    # Simulated feature importance
    diseases = ['CAD', 'Stroke', 'CKD', 'T2D', 'Depression', 'AD']
    
    top_predictors = {}
    for disease in diseases:
        # Simulated feature importance (would come from model.get_booster().get_score())
        predictors = [
            {"feature": "Age", "importance": 0.28, "direction": "positive"},
            {"feature": "SBP", "importance": 0.25, "direction": "positive"},
            {"feature": "SBP_PRS", "importance": 0.18, "direction": "positive"},
            {"feature": "BMI", "importance": 0.15, "direction": "positive"},
            {"feature": "Smoking", "importance": 0.10, "direction": "positive"},
            {"feature": "DBP_PRS", "importance": 0.04, "direction": "positive"}
        ]
        top_predictors[disease] = predictors
    
    return top_predictors

def get_clinical_actions(individual_risks: Dict[str, RiskPrediction]) -> Dict[str, List[str]]:
    """Get recommended clinical actions based on risk levels"""
    actions = {}
    
    for disease, prediction in individual_risks.items():
        risk_level = prediction.risk_category
        
        if disease in clinical_mapping:
            disease_actions = clinical_mapping[disease]
            if risk_level in disease_actions:
                actions[disease] = disease_actions[risk_level]
            else:
                actions[disease] = disease_actions.get('General', [])
        else:
            actions[disease] = ["Consult physician for personalized risk assessment"]
    
    return actions

@app.get("/", response_model=Dict)
async def root():
    """API root endpoint"""
    return {
        "message": "MMRP Clinical AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(model_config) > 0
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PatientFeatures, patient_id: Optional[str] = None):
    """
    Predict comorbidity risks for a single patient
    
    Returns individual disease risks, MMRS score, and clinical recommendations
    """
    try:
        logger.info(f"Processing prediction request for patient: {patient_id or 'anonymous'}")
        
        # Predict for all diseases
        diseases = ['CAD', 'Stroke', 'CKD', 'T2D', 'Depression', 'AD']
        individual_risks = {}
        
        for disease in diseases:
            individual_risks[disease] = predict_disease_risk(features, disease)
        
        # Calculate MMRS
        mmrs_score, mmrs_category = calculate_mmrs(individual_risks)
        
        # Get top predictors
        top_predictors = get_top_predictors(features)
        
        # Get clinical actions
        clinical_actions = get_clinical_actions(individual_risks)
        
        response = PredictionResponse(
            patient_id=patient_id,
            timestamp=datetime.now().isoformat(),
            individual_risks=individual_risks,
            mmrs_score=mmrs_score,
            mmrs_category=mmrs_category,
            top_predictors=top_predictors,
            clinical_actions=clinical_actions
        )
        
        logger.info(f"Prediction completed successfully. MMRS: {mmrs_score}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple patients
    
    Efficiently processes multiple patients in one request
    """
    try:
        logger.info(f"Processing batch prediction for {len(request.patients)} patients")
        
        responses = []
        for i, patient_features in enumerate(request.patients):
            patient_id = request.patient_ids[i] if request.patient_ids and i < len(request.patient_ids) else f"patient_{i+1}"
            
            # Reuse single prediction logic
            response = await predict(patient_features, patient_id)
            responses.append(response)
        
        logger.info(f"Batch prediction completed for {len(responses)} patients")
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "version": "1.0.0",
        "models_loaded": len(model_config) > 0,
        "diseases": ['CAD', 'Stroke', 'CKD', 'T2D', 'Depression', 'AD'],
        "features": [
            "SBP_PRS", "DBP_PRS", "PP_PRS", "Age", "Sex", "BMI",
            "SBP", "Smoking", "Physical_Activity", "Diet_Score"
        ],
        "config": model_config
    }

@app.get("/features/schema")
async def get_feature_schema():
    """Get feature schema and validation rules"""
    return {
        "features": {
            "sbp_prs": {"type": "float", "min": -5, "max": 5, "description": "SBP Polygenic Risk Score"},
            "dbp_prs": {"type": "float", "min": -5, "max": 5, "description": "DBP Polygenic Risk Score"},
            "pp_prs": {"type": "float", "min": -5, "max": 5, "description": "Pulse Pressure PRS"},
            "age": {"type": "float", "min": 30, "max": 85, "description": "Patient age in years"},
            "sex": {"type": "int", "min": 0, "max": 1, "description": "Sex (0=Female, 1=Male)"},
            "bmi": {"type": "float", "min": 15, "max": 50, "description": "Body Mass Index"},
            "sbp": {"type": "float", "min": 90, "max": 200, "description": "Systolic Blood Pressure"},
            "smoking": {"type": "int", "min": 0, "max": 1, "description": "Smoking status"},
            "physical_activity": {"type": "float", "min": 0, "max": 10, "description": "Physical activity score"},
            "diet_score": {"type": "float", "min": 0, "max": 10, "description": "Diet quality score"}
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
