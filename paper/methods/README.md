# HPCMA Methods Supplement

## Detailed Methodological Protocols

This document provides comprehensive methodological details for the Hypertension Pan-Comorbidity Multi-Modal Atlas (HPCMA).

---

## Table of Contents

1. [Step 1: GWAS Harmonization & Quality Control](#step-1-gwas-harmonization--quality-control)
2. [Step 2: Genetic Architecture Analysis](#step-2-genetic-architecture-analysis)
3. [Step 3: Causal Gene Prioritization](#step-3-causal-gene-prioritization)
4. [Step 4: Cell-Type Mapping](#step-4-cell-type-mapping)
5. [Step 5: Multi-Modal Prediction Models](#step-5-multi-modal-prediction-models)
6. [Step 6: Atlas Integration](#step-6-atlas-integration)
7. [Step 7: Validation](#step-7-validation)

---

## Step 1: GWAS Harmonization & Quality Control

### 1.1 Dataset Selection Criteria

**Inclusion Criteria**:
- Sample size > 10,000 individuals
- European ancestry (95%+ for primary atlas)
- Genome-wide coverage (>500,000 SNPs post-QC)
- Published in peer-reviewed journal or established consortium
- Available summary statistics

**Datasets Harmonized**:

| Trait | Dataset ID | Source | N | Build |
|-------|-----------|--------|---|-------|
| SBP | UKB-IEU | UK Biobank | 317,754 | GRCh37 |
| DBP | UKB-IEU | UK Biobank | 317,754 | GRCh37 |
| PP | UKB-IEU | UK Biobank | 317,754 | GRCh37 |
| CAD | CARDIoGRAM+C4D | Consortium | 184,305 | GRCh37 |
| Stroke | MEGASTROKE | Consortium | 514,791 | GRCh37 |
| CKD | CKDGen | Consortium | 117,480 | GRCh37 |
| T2D | DIAGRAM | Consortium | 159,208 | GRCh37 |
| Depression | PGC | Consortium | 500,199 | GRCh37 |
| AD | IGAP | Consortium | 63,926 | GRCh37 |

### 1.2 Harmonization Pipeline

```bash
# Tool: GWAS Harmonization Pipeline v2.0
# Reference: Haplotype Reference Consortium (HRC) panel

# 1. Genome build alignment
liftOver input.b37.gz hg19ToHg38.over.chain output.b38.gz

# 2. Allele frequency harmonization
# Match to 1000 Genomes EUR super-population
# Exclude SNPs with AF difference > 0.20

# 3. Effect allele standardization
# Orient all effects to ALT allele
# Require consistent allele coding

# 4. Quality control filters
# - Remove indels
# - Remove SNPs with INFO < 0.8
# - Remove palindromic SNPs (A/T, C/G)
# - Remove SNPs with SE > 10
```

### 1.3 Quality Control Metrics

**Per-Dataset QC**:

```python
# Minimum thresholds
QC_CRITERIA = {
    'sample_size': 10000,
    'snp_count_post_qc': 500000,
    'mean_se': 0.5,  # Maximum allowed
    'lambda_gc': [0.95, 1.15],  # Acceptable range
    'allele_missing_rate': 0.05  # Maximum
}

# All 11 datasets passed QC
```

---

## Step 2: Genetic Architecture Analysis

### 2.1 LD Score Regression

**Software**: LDSC v1.0.1

**Command**:
```bash
python ldsc.py \
    --rg trait1.sumstats.gz,trait2.sumstats.gz \
    --ref-ld-chr eur_w_ld_chr/ \
    --w-ld-chr eur_w_ld_chr/ \
    --out genetic_correlation \
    --intercept-h2 \
    --no-intercept-rg
```

**Parameters**:
- Reference panel: 1000 Genomes EUR
- LD windows: 1 cM
- Heritability model: BaselineLD v2.2

### 2.2 Genetic Correlation Estimation

**Significance Testing**:
```python
# Z-score calculation
z = rg / se

# Two-tailed p-value
p = 2 * (1 - norm.cdf(abs(z)))

# Multiple testing: Bonferroni correction
n_tests = 55  # 11 traits choose 2
alpha_corrected = 0.05 / n_tests
```

### 2.3 Cross-Trait Meta-Analysis

**Method**: MT-COJO (Multi-Trait COnditional & JOint analysis)

**Procedure**:
1. Identify lead SNPs for each trait (p < 5e-8)
2. Test for shared associations
3. Calculate shared loci count per pair

---

## Step 3: Causal Gene Prioritization

### 3.1 Mendelian Randomization

**Software**: TwoSampleMR v0.5.6 (R package)

**Instrument Selection**:
```r
# Clumping parameters
clump_r2 <- 0.01
clump_kb <- 10000
clump_p <- 5e-8

# F-statistic threshold
F_threshold <- 10

# Instrument extraction
instruments <- extract_instruments(
    outcomes = exposure_id,
    p1 = clump_p,
    clump = TRUE,
    r2 = clump_r2,
    kb = clump_kb,
    access_token = NULL
)

# F-statistic calculation
F_stat <- (beta^2) / (se^2)
instruments <- instruments[F_stat >= F_threshold, ]
```

**MR Methods**:

1. **Inverse Variance Weighted (IVW)**
   - Primary method for main estimates
   - Assumes no pleiotropy or balanced pleiotropy

2. **MR-Egger**
   - Allows directional pleiotropy
   - Test via intercept (should not differ from zero)

3. **Weighted Median**
   - Robust to 50% invalid instruments
   - Used for sensitivity analysis

4. **Weighted Mode**
   - Uses mode of Wald ratios
   - Assumes plurality valid

**Sensitivity Analyses**:
- Leave-one-out analysis
- Funnel plot asymmetry assessment
- MR-PRESSO outlier detection

### 3.2 Colocalization Analysis

**Software**: coloc v5.2.0 (R package)

**Method**: Bayesian colocalization

```r
# Coloc analysis
coloc_result <- coloc.abf(
    dataset1 = list(
        pvalues = pval_trait1,
        N = sample_size_trait1,
        beta = beta_trait1,
        varbeta = se_trait1^2,
        type = "quant"
    ),
    dataset2 = list(
        pvalues = pval_trait2,
        N = sample_size_trait2,
        beta = beta_trait2,
        varbeta = se_trait2^2,
        type = "cc"  # case-control
    ),
    MAF = maf_vector,
    p1 = 1e-4,   # Prior for trait1
    p2 = 1e-4,   # Prior for trait2
    p12 = 1e-5   # Prior for shared variant
)

# Interpretation
PPH4 <- coloc_result$summary["PP.H4.abf"]
# PPH4 > 0.75: High confidence shared causal variant
# PPH4 0.5-0.75: Moderate confidence
# PPH4 < 0.5: Low confidence
```

**Genomic Window**: ±100 kb around index SNP

### 3.3 Evidence Integration

**Priority Score Calculation**:

```python
def calculate_priority_score(gene):
    score = 0
    
    # MR evidence (max 40 points)
    if gene.has_significant_mr:
        score += min(abs(gene.mr_beta) * 20, 40)
    
    # Colocalization evidence (max 30 points)
    if gene.max_pph4 > 0.75:
        score += 30
    elif gene.max_pph4 > 0.5:
        score += 20
    elif gene.max_pph4 > 0.25:
        score += 10
    
    # eQTL evidence (max 20 points)
    if gene.has_eqtl_support:
        score += 20
    
    # Biological plausibility (max 10 points)
    if gene.in_pathway:
        score += 10
    
    return score

# Tier assignment
# Tier 1: Score >= 80
# Tier 2: Score 60-79
# Tier 3: Score 40-59
```

---

## Step 4: Cell-Type Mapping

### 4.1 Single-Cell Reference Datasets

**Primary Atlases**:

| Atlas | Source | Tissues | Cells | Version |
|-------|--------|---------|-------|---------|
| Human Cell Atlas | HCA | Multi-organ | 500K | v1.0 |
| GTEx | GTEx Portal | 54 tissues | 100K | v8 |
| Tabula Sapiens | TS | Multi-organ | 400K | v1.0 |
| Heart Cell Atlas | HCA | Cardiac | 50K | v1.0 |

### 4.2 Cell-Type Specificity Analysis

**Tau Calculation** (tissue/cell specificity index):

```python
import numpy as np

def calculate_tau(expression_vector):
    """
    Tau ranges from 0 (ubiquitous) to 1 (specific)
    """
    x = expression_vector / np.max(expression_vector)
    tau = np.sum(1 - x) / (len(x) - 1)
    return tau

# Interpretation
# Tau > 0.8: Highly specific
# Tau 0.5-0.8: Moderately specific
# Tau < 0.5: Broadly expressed
```

**Specificity Score**:

```python
def cell_type_specificity_score(gene_expr, cell_type_expr, all_cells_expr):
    """
    Compare expression in cell type vs all other cells
    """
    cell_mean = np.mean(gene_expr[cell_type_mask])
    other_mean = np.mean(gene_expr[~cell_type_mask])
    
    specificity = (cell_mean - other_mean) / (cell_mean + other_mean)
    return specificity
```

### 4.3 Mechanism Axis Assignment

**Axis Definitions**:

1. **Vascular Tone Regulation**
   - Key genes: ACE, AGT, EDN1, NOS3
   - Cell types: Endothelial, Vascular smooth muscle
   - Function: Blood pressure regulation via vascular resistance

2. **Renal Salt Handling**
   - Key genes: SHROOM3, UMOD
   - Cell types: Podocytes, Tubular epithelium
   - Function: Sodium reabsorption and volume regulation

3. **Cardiac Natriuretic Signaling**
   - Key genes: NPPA
   - Cell types: Cardiomyocytes
   - Function: Cardiac remodeling and pressure-volume regulation

**Assignment Criteria**:
- Must have Tau > 0.5 in disease-relevant tissue
- Mechanism score > 0.6
- Literature-supported functional role

---

## Step 5: Multi-Modal Prediction Models

### 5.1 Feature Engineering

**Genetic Features** (3):
- PRS_SBP: Polygenic risk score for systolic BP
- PRS_DBP: Polygenic risk score for diastolic BP
- PRS_PP: Polygenic risk score for pulse pressure

**Clinical Features** (4):
- Age: Years (30-85)
- Sex: 0=Male, 1=Female
- BMI: kg/m² (15-50)
- Hypertension_Status: 0/1

**Environmental Features** (3):
- Smoking_Status: 0=No, 1=Yes
- Salt_Intake: grams/day (0-15)
- Physical_Activity: minutes/week (0-300)

**Total**: 10 features

### 5.2 Model Architectures

#### XGBoost (Primary)

```python
import xgboost as xgb

# Hyperparameters (tuned via 5-fold CV)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 3,
    'gamma': 0.1
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# SHAP explainability
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

#### Random Forest (Ensemble)

```python
from sklearn.ensemble import RandomForestClassifier

rf_params = {
    'n_estimators': 500,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced'
}

rf_model = RandomForestClassifier(**rf_params)
```

#### Logistic Regression (Baseline)

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced'
)
```

### 5.3 Training Protocol

**Cross-Validation**: 5-fold stratified CV

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train and evaluate
    model.fit(X_train_fold, y_train_fold)
    fold_auc = roc_auc_score(y_val_fold, model.predict_proba(X_val_fold)[:, 1])
```

**Ensemble Strategy**:

```python
# Weighted average ensemble
ensemble_proba = (
    0.5 * xgb_proba + 
    0.3 * rf_proba + 
    0.2 * lr_proba
)
```

### 5.4 Model Evaluation

**Metrics**:
- AUC-ROC (primary)
- AUC-PR (for imbalanced classes)
- Calibration (slope, intercept)
- Decision curve analysis

**Thresholds**:
- Clinical QC threshold: AUC ≥ 0.60
- High performance: AUC ≥ 0.80

---

## Step 6: Atlas Integration

### 6.1 Multi-Layer Network Construction

**Layers**:
1. Gene-Disease (causal relationships)
2. Gene-Cell Type (expression specificity)
3. Cell Type-Disease (pathological role)

**Network Edges**:

```python
edges = []

# Layer 1: Gene-Disease
for gene, disease in significant_mr_pairs:
    edges.append({
        'Layer': 'Gene_Disease',
        'Source': gene,
        'Target': disease,
        'Weight': abs(mr_beta),
        'Evidence': 'MR'
    })

# Layer 2: Gene-Cell Type
for gene, cell_type in specific_expression:
    edges.append({
        'Layer': 'Gene_Cell',
        'Source': gene,
        'Target': cell_type,
        'Weight': tau_score,
        'Evidence': 'Expression'
    })

# Layer 3: Cell Type-Disease
for cell_type, disease in disease_associations:
    edges.append({
        'Layer': 'Cell_Disease',
        'Source': cell_type,
        'Target': disease,
        'Weight': enrichment_pvalue,
        'Evidence': 'Literature'
    })
```

### 6.2 Clinical Translation Mapping

**Risk Score Interpretation**:

| Individual Probability | Risk Category | Clinical Action |
|----------------------|---------------|-----------------|
| < 0.15 | Low | Routine screening |
| 0.15-0.30 | Moderate | Lifestyle counseling |
| 0.30-0.45 | High | Active intervention |
| > 0.45 | Very High | Aggressive management |

**MMRS Composite Score**:

```python
def calculate_mmrs(individual_risks, weights=None):
    if weights is None:
        weights = [1, 1, 1, 1, 1, 1]  # Equal weights
    
    mmrs = np.average(individual_risks, weights=weights)
    return mmrs

# Categories
# MMRS < 0.20: Low Risk
# MMRS 0.20-0.35: Moderate Risk
# MMRS 0.35-0.50: High Risk
# MMRS > 0.50: Very High Risk
```

---

## Step 7: Validation

### 7.1 Internal Validation

**Bootstrap Stability**:
```python
from sklearn.utils import resample

n_bootstrap = 100
stability_scores = []

for i in range(n_bootstrap):
    X_boot, y_boot = resample(X_train, y_train)
    model.fit(X_boot, y_boot)
    
    # Calculate stability metric
    stability = compare_with_original(model, original_model)
    stability_scores.append(stability)

# Report mean and 95% CI
```

**Cross-Validation Performance**:
- 5-fold stratified CV
- Report mean ± SD across folds
- Check for overfitting (train vs test gap < 0.05 AUC)

### 7.2 External Validation

**Protocol**:
1. Independent cohort (no sample overlap)
2. Same feature definitions
3. Same model (frozen weights)
4. Prospective prediction

**PRS Shift Test**:
```python
# Compare PRS distributions
discovery_mean = np.mean(prs_discovery)
validation_mean = np.mean(prs_validation)

# Two-sample t-test
t_stat, p_value = ttest_ind(prs_discovery, prs_validation)

# Interpretation
if p_value < 0.05:
    print("Warning: PRS distribution shifted")
else:
    print("PRS portable to validation cohort")
```

### 7.3 Bias Assessment

**Demographic Stratification**:

```python
# Performance by sex
male_auc = roc_auc_score(y_male, pred_male)
female_auc = roc_auc_score(y_female, pred_female)

# Performance by age group
age_bins = [30, 45, 60, 75, 85]
for i in range(len(age_bins)-1):
    mask = (age >= age_bins[i]) & (age < age_bins[i+1])
    group_auc = roc_auc_score(y[mask], pred[mask])
```

**Fairness Metrics**:
- Demographic parity: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
- Equalized odds: TPR and FPR equal across groups
- Calibration: Risk scores well-calibrated within groups

---

## Software Versions

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9.0 | Main language |
| pandas | 2.0.0 | Data manipulation |
| numpy | 1.24.0 | Numerical computing |
| scipy | 1.11.0 | Statistical tests |
| scikit-learn | 1.3.0 | Machine learning |
| xgboost | 2.0.0 | Gradient boosting |
| shap | 0.42.0 | Model explainability |
| LDSC | 1.0.1 | Genetic correlations |
| TwoSampleMR | 0.5.6 | Mendelian randomization |
| coloc | 5.2.0 | Colocalization |
| Seurat | 4.3.0 | Single-cell analysis |
| Scanpy | 1.9.0 | Single-cell analysis |

---

## Computational Environment

**Hardware**:
- CPU: Intel Xeon E5-2680 v4 (14 cores)
- RAM: 128 GB
- Storage: 2 TB SSD

**Runtime Estimates**:

| Step | Task | Runtime (single core) | Parallelized |
|------|------|---------------------|--------------|
| 1 | GWAS harmonization (11 datasets) | ~2 hours | Yes (11 parallel) |
| 2 | LDSC genetic correlation | ~4 hours | Yes (chromosomes) |
| 3 | MR analysis (18 pairs) | ~1 hour | Yes |
| 3 | Colocalization (13 loci) | ~30 min | Yes |
| 4 | Cell-type mapping | ~2 hours | Yes (tissues) |
| 5 | ML model training | ~30 min | Yes (diseases) |
| 6 | Atlas integration | ~15 min | No |
| 7 | Validation | ~20 min | No |
| **Total** | | **~10 hours** | **~2 hours (parallel)** |

---

## References

Key methodological papers:

1. Bulik-Sullivan et al. (2015) LD Score Regression
2. Hemani et al. (2018) TwoSampleMR
3. Giambartolomei et al. (2014) coloc
4. Chen & Guestrin (2016) XGBoost
5. Liberzon et al. (2015) Molecular Signatures Database

---

**Document Version**: 1.0.0  
**Last Updated**: February 2025  
**Maintained by**: Benjamin-JHou
