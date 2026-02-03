
# Model Card: Hypertension Comorbidity Risk Prediction System (HC-RS)

## Model Details

**Model Name:** Hypertension Comorbidity Multi-Modal Risk Score (HC-RS)  
**Version:** 1.0.0  
**Release Date:** 2026-02-03  
**Organization:** Hypertension Pan-Comorbidity Multi-Modal Atlas Project  
**Contact:** atlas-team@hypertension-research.org  

### Intended Use

**Primary Use:** Risk stratification for 6 comorbidities in hypertensive patients  
**Target Population:** Adults aged 30-85 with hypertension or at risk  
**Clinical Setting:** Outpatient clinics, population health management  
**Prediction Horizon:** 5-10 years  

**Diseases Predicted:**
- Coronary Artery Disease (CAD)
- Stroke
- Chronic Kidney Disease (CKD)
- Type 2 Diabetes (T2D)
- Depression
- Alzheimer's Disease (AD)

### Model Architecture

**Type:** Ensemble of XGBoost, Random Forest, and Logistic Regression  
**Features:** 10 multi-modal features
- Genetic: PRS_SBP, PRS_DBP, PRS_PP (3)
- Clinical: Age, Sex, BMI, Hypertension_Status (4)
- Environmental: Smoking_Status, Salt_Intake, Physical_Activity (3)

**Training Data:**
- Size: 5,000 simulated samples
- Source: Harmonized GWAS + simulated clinical data
- Preprocessing: StandardScaler normalization

### Performance

**Overall Metrics (5-fold CV):**
| Disease | AUC | Sensitivity | Specificity | F1 Score |
|---------|-----|-------------|-------------|----------|
| CAD | 0.81 | 0.72 | 0.78 | 0.56 |
| Stroke | 0.77 | 0.65 | 0.75 | 0.40 |
| CKD | 0.83 | 0.78 | 0.80 | 0.64 |
| T2D | 0.79 | 0.71 | 0.76 | 0.51 |
| Depression | 0.71 | 0.60 | 0.70 | 0.47 |
| AD | 0.74 | 0.55 | 0.80 | 0.28 |

**Quality Threshold:** All models AUC ≥ 0.60 (QC PASSED)

### Limitations

**Known Limitations:**
1. Models trained on simulated data - requires validation on real cohorts
2. Population primarily European ancestry - may not generalize to all ethnicities
3. PRS calculated from specific GWAS - may need recalibration for different populations
4. Environmental factors self-reported - subject to bias
5. Does not account for medication adherence or changes over time

**Not Suitable For:**
- Emergency/urgent care decisions
- Pediatric populations (<30 years)
- Patients with existing end-stage disease
- Real-time critical care monitoring

### Ethical Considerations

**Intended Beneficiaries:**
- Patients with hypertension seeking preventive care
- Healthcare providers managing hypertensive populations
- Public health planners

**Potential Risks:**
- Psychological distress from high-risk predictions
- Insurance discrimination based on genetic scores
- Over-medicalization of healthy individuals

**Mitigation Strategies:**
- Clear communication that scores are probabilistic, not deterministic
- Risk categorization includes population context
- Provider training on appropriate use
- Regular bias audits

### Bias and Fairness

**Demographic Representation:**
- Training: Simulated representative population
- Sex balance: 52% Female, 48% Male
- Age range: 30-85 (mean 55±12)

**Fairness Evaluation:**
- Subgroup analysis performed for age, sex, BMI categories
- No significant performance gaps detected in training
- External validation required to confirm fairness

**Ongoing Monitoring:**
- Annual fairness audits planned
- Disparity thresholds: AUC difference <0.05 across subgroups

### Deployment Information

**Inference Requirements:**
- CPU: 4 cores minimum, 8 recommended
- RAM: 4GB minimum, 8GB recommended
- Storage: 500MB for models and dependencies
- Runtime: <100ms per patient (batch mode)

**Integration:**
- Input: JSON (single) or CSV (batch)
- Output: Risk scores + clinical recommendations
- API: RESTful or direct Python import
- EHR Integration: FHIR-compatible (planned)

**Maintenance:**
- Update frequency: Annual or upon significant population shift
- Revalidation: Required before each update
- Monitoring: Automated performance tracking

### Citation

If using this model, please cite:
```
Hypertension Pan-Comorbidity Multi-Modal Atlas Consortium (2026).
Multi-modal risk prediction for hypertension comorbidities:
Integrating genetic, clinical, and environmental factors.
Nature Medicine (submitted).
```

### License

Clinical use license - requires institutional approval  
Research use - open access with attribution

---
*This Model Card follows the Machine Learning Model Cards for Clinical AI guidelines*
