"""
Test suite for MMRP Clinical AI API
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.inference.api_server import app, PatientFeatures

client = TestClient(app)

class TestHealthEndpoints:
    """Test health and info endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "MMRP Clinical AI API" in data["message"]
        assert data["version"] == "1.0.0"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_models_info(self):
        """Test models info endpoint"""
        response = client.get("/models/info")
        assert response.status_code == 200
        data = response.json()
        assert "diseases" in data
        assert len(data["diseases"]) == 6
        assert "features" in data
    
    def test_feature_schema(self):
        """Test feature schema endpoint"""
        response = client.get("/features/schema")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "sbp_prs" in data["features"]

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_single_prediction(self):
        """Test single patient prediction"""
        patient_data = {
            "sbp_prs": 0.5,
            "dbp_prs": 0.3,
            "pp_prs": 0.2,
            "age": 55,
            "sex": 1,
            "bmi": 28,
            "sbp": 140,
            "smoking": 0,
            "physical_activity": 5,
            "diet_score": 6
        }
        
        response = client.post("/predict?patient_id=test_patient_001", json=patient_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["patient_id"] == "test_patient_001"
        assert "individual_risks" in data
        assert "mmrs_score" in data
        assert "mmrs_category" in data
        assert "clinical_actions" in data
        assert len(data["individual_risks"]) == 6
        
        # Check risk values are in valid range
        for disease, risk in data["individual_risks"].items():
            assert 0 <= risk["probability"] <= 1
            assert risk["risk_category"] in ["Low", "Moderate", "High", "Very High"]
    
    def test_prediction_validation(self):
        """Test input validation"""
        # Invalid age
        invalid_data = {
            "sbp_prs": 0.5,
            "dbp_prs": 0.3,
            "pp_prs": 0.2,
            "age": 150,  # Invalid
            "sex": 1,
            "bmi": 28,
            "sbp": 140,
            "smoking": 0,
            "physical_activity": 5,
            "diet_score": 6
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        batch_data = {
            "patients": [
                {
                    "sbp_prs": 0.5, "dbp_prs": 0.3, "pp_prs": 0.2,
                    "age": 55, "sex": 1, "bmi": 28, "sbp": 140,
                    "smoking": 0, "physical_activity": 5, "diet_score": 6
                },
                {
                    "sbp_prs": -0.2, "dbp_prs": -0.1, "pp_prs": 0.1,
                    "age": 45, "sex": 0, "bmi": 24, "sbp": 125,
                    "smoking": 0, "physical_activity": 8, "diet_score": 8
                }
            ],
            "patient_ids": ["patient_001", "patient_002"]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 2
        assert data[0]["patient_id"] == "patient_001"
        assert data[1]["patient_id"] == "patient_002"

class TestRiskCalculations:
    """Test risk calculation logic"""
    
    def test_mmrs_calculation(self):
        """Test MMRS score calculation"""
        patient_data = {
            "sbp_prs": 1.5,  # High genetic risk
            "dbp_prs": 1.2,
            "pp_prs": 1.0,
            "age": 70,  # High age
            "sex": 1,
            "bmi": 32,  # High BMI
            "sbp": 160,  # High BP
            "smoking": 1,  # Smoker
            "physical_activity": 2,  # Low activity
            "diet_score": 3  # Poor diet
        }
        
        response = client.post("/predict?patient_id=high_risk", json=patient_data)
        assert response.status_code == 200
        
        data = response.json()
        # Should have elevated risk
        assert data["mmrs_score"] > 0.4
        assert data["mmrs_category"] in ["High Risk", "Very High Risk"]
    
    def test_low_risk_calculation(self):
        """Test low risk calculation"""
        patient_data = {
            "sbp_prs": -1.0,  # Low genetic risk
            "dbp_prs": -0.8,
            "pp_prs": -0.5,
            "age": 35,  # Young
            "sex": 0,
            "bmi": 22,  # Normal BMI
            "sbp": 115,  # Normal BP
            "smoking": 0,  # Non-smoker
            "physical_activity": 9,  # High activity
            "diet_score": 9  # Good diet
        }
        
        response = client.post("/predict?patient_id=low_risk", json=patient_data)
        assert response.status_code == 200
        
        data = response.json()
        # Should have low risk
        assert data["mmrs_score"] < 0.25
        assert data["mmrs_category"] == "Low Risk"

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_batch(self):
        """Test empty batch prediction"""
        batch_data = {
            "patients": [],
            "patient_ids": []
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        assert response.json() == []
    
    def test_boundary_values(self):
        """Test boundary value inputs"""
        boundary_data = {
            "sbp_prs": 5.0,  # Maximum
            "dbp_prs": -5.0,  # Minimum
            "pp_prs": 0.0,
            "age": 85,  # Maximum age
            "sex": 1,
            "bmi": 50,  # Maximum BMI
            "sbp": 200,  # Maximum BP
            "smoking": 1,
            "physical_activity": 10,  # Maximum
            "diet_score": 10  # Maximum
        }
        
        response = client.post("/predict", json=boundary_data)
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
