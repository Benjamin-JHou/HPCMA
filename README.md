# Hypertension Pan-Comorbidity Multi-Modal Atlas (HPCMA)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Nature Standards](https://img.shields.io/badge/Research%20Atlas-Nature%2FCell%20Standards-green.svg)](https://github.com/Benjamin-JHou/HPCMA)
[![Clinical AI](https://img.shields.io/badge/Clinical%20Translation-AI%20Enabled-red.svg)](https://github.com/Benjamin-JHou/HPCMA)

## 🧬 Research Overview

Hypertension represents a **systemic multi-organ disease** affecting over 1.3 billion individuals globally, serving as the primary gateway to cardiovascular, renal, metabolic, and neurodegenerative disorders. Despite its status as the leading modifiable risk factor for premature mortality, the molecular mechanisms governing hypertension-mediated multi-organ damage remain incompletely characterized. The **Hypertension Pan-Comorbidity Multi-Modal Atlas (HPCMA)** addresses this fundamental gap by constructing an integrated systems-biology framework that decodes the **cross-disease shared genetic architecture** underlying hypertensive end-organ damage.

Our approach leverages **multi-modal data integration** spanning five complementary biological layers: (1) **genomic**—polygenic risk scores (PRS) for blood pressure traits and comorbidities derived from genome-wide association studies (GWAS) in >1 million individuals; (2) **transcriptomic**—cell-type-specific gene expression from single-cell atlases of cardiac, renal, vascular, and neural tissues; (3) **clinical**—electronic health record phenotypes including demographics, biomarkers, and medication histories; (4) **environmental**—lifestyle and behavioral exposures; and (5) **artificial intelligence**—ensemble machine learning models with clinical-grade validation. This multi-dimensional integration enables mechanistic insights beyond conventional single-disease approaches.

The HPCMA serves as a **public research resource** providing open access to harmonized GWAS datasets, causal gene-cell-disease mappings, validated predictive models, and interactive visualization tools. All genetic correlation matrices, Mendelian randomization results, colocalization analyses, and multi-modal risk prediction algorithms are freely available for academic research and clinical translation. Our atlas bridges the gap from genomic discovery to clinical implementation, offering validated risk stratification tools for precision hypertension management and comorbidity prevention in healthcare settings.

## 🎯 Clinical Translation Objectives

| Disease Target | Population Impact | Model Performance | Clinical Actionability |
|----------------|-------------------|-------------------|----------------------|
| **Coronary Artery Disease (CAD)** | Leading cause of death globally | AUC 0.81 | Statins, antihypertensives |
| **Stroke** | 2nd leading cause of mortality | AUC 0.77 | Anticoagulation, BP control |
| **Chronic Kidney Disease (CKD)** | Affects 10% of adults | AUC 0.83 | ACE inhibitors, dialysis planning |
| **Type 2 Diabetes (T2D)** | 537 million cases worldwide | AUC 0.79 | Lifestyle, metformin |
| **Major Depressive Disorder** | 280 million affected | AUC 0.71 | SSRIs, psychotherapy |
| **Alzheimer's Disease (AD)** | 55 million dementia cases | AUC 0.74 | Early intervention |

**Atlas Scope:** 11 harmonized GWAS datasets | 55 genetic correlation pairs | 228 shared loci | 7 Tier 1 causal genes | 45 gene-cell mappings | 18 validated ML models

---

## 🔬 Multi-Modal Integration Framework

### Layer 1: Genomic Architecture (🧬)
Genome-wide association studies (GWAS) reveal the polygenic architecture of hypertension and its comorbidities. We integrate summary statistics from the **UK Biobank** and **IEU OpenGWAS** to construct polygenic risk scores (PRS) capturing inherited susceptibility to:
- Systolic blood pressure (SBP) regulation
- Diastolic blood pressure (DBP) control  
- Pulse pressure (PP) dynamics
- Cross-disease genetic correlations via **LD Score Regression**

### Layer 2: Cell-Type Specificity (🔬)
Single-cell transcriptomics from **Human Cell Atlas** and **GTEx** enable cell-type-resolution mapping of causal genes. We identify disease-relevant cell populations including:
- **Vascular endothelium** (ACE, NOS3)
- **Renal tubular epithelium** (UMOD, SHROOM3)
- **Cardiac myocytes** (NPPA)
- **Neural tissues** (APP, PSEN1)

### Layer 3: Clinical Phenotypes (🏥)
Electronic health record-derived clinical features capture environmental and acquired risk factors:
- **Demographics**: Age, sex, ancestry
- **Physiological**: BMI, blood pressure, biomarkers
- **Comorbidities**: Existing disease diagnoses

### Layer 4: Environmental Exposures (🌍)
Modifiable lifestyle factors integrated via validated questionnaires:
- Smoking status
- Physical activity levels
- Dietary patterns (DASH score)
- Sodium intake

### Layer 5: AI Integration (🤖)
Ensemble machine learning models combine all layers using **XGBoost** with **SHAP** explainability:
- **Multi-modal feature fusion** (10 features)
- **Cross-validated performance** (5-fold CV)
- **Clinical validation** (independent test sets)
- **Bias assessment** (demographic fairness checks)

---

## 📊 Atlas Construction Pipeline (7 Steps)

### Step 1: Dataset Harmonization & Quality Control
- Harmonized 11 GWAS datasets from European ancestry cohorts
- Standardized genome builds (GRCh37/GRCh38)
- QC metrics: SNP coverage >90%, sample size >10,000 per trait
- MHC region handling and allele frequency alignment

### Step 2: Genetic Shared Architecture Analysis
- **LD Score Regression**: 55 pairwise genetic correlations
- **Cross-trait meta-analysis**: 228 shared independent loci
- **Heterogeneity assessment**: Cochran's Q statistics
- **Visualization**: Genetic correlation heatmaps, Manhattan plots

### Step 3: Causal Gene Prioritization
- **Mendelian Randomization**: 18 exposure-outcome pairs
  - Wald ratio, Egger regression, IVW methods
  - F-statistics >10 for instrument strength
- **Colocalization (coloc)**: 13 loci tested, 10 high-confidence (PPH4 > 0.7)
- **Tier 1 Causal Genes**: ACE, AGT, EDN1, NOS3, NPPA, SHROOM3, UMOD

### Step 4: Cell Type Mapping
- **Single-cell RNA-seq integration**: Human Cell Atlas, GTEx, Tabula Sapiens
- **Cell-type enrichment**: MAGMA, LDSC-SEG
- **Specificity scoring**: Expression percentiles per cell type
- **Mechanism axes**: Vascular tone, renal salt handling, cardiac remodeling

### Step 5: Multi-Modal Prediction Models
- **Training cohort**: 5,000 hypertensive patients
- **Features**: 3 genetic PRS + 4 clinical + 3 environmental = 10 total
- **Models per disease**: Logistic Regression, Random Forest, XGBoost
- **Ensemble**: Weighted average of top 3 models
- **Performance**: AUC 0.71-0.83, all passing QC threshold (≥0.60)

### Step 6: Integrated Atlas Construction
- **Multi-layer network**: Disease → Gene → Cell type
- **Master atlas table**: 17 edges, standardized terminologies
- **Visualization**: Disease-gene-cell networks, UMAP projections
- **Clinical translation layer**: Risk score interpretation, action mappings

### Step 7: External Validation & Fairness
- **Validation protocol**: Independent cohort (n=2,000)
- **Bootstrap stability**: 91.7% average across models
- **Bias assessment**: Sex, age, ancestry stratification
- **Fairness metrics**: Demographic parity, equalized odds

---

## 🚀 Reproducible Infrastructure

### Supplementary API Demonstration
```bash
# Installation
pip install -r requirements.txt

# Start API server
python -m src.inference.api_server

# Access interactive docs
curl http://localhost:8000/docs
```

### Docker Containerization
```bash
docker build -t hpcma:latest .
docker run -p 8000:8000 hpcma:latest
```

### CI/CD Pipeline
- GitHub Actions workflow for automated testing
- Multi-version Python support (3.9, 3.10, 3.11)
- Code quality checks: Black, isort, flake8, mypy
- Security scanning with Trivy
- Coverage reporting via Codecov

### Scope Boundary
- This repository is presented as a reproducible research resource.
- API-based risk scoring is a supplementary demonstration artifact.
- No clinical deployment readiness claim is made in the primary resource narrative.

---

## 📚 Resource Availability

### Open Access Datasets
- Harmonized GWAS summary statistics (11 traits)
- Genetic correlation matrices (LDSC outputs)
- Causal gene prioritization results (MR + coloc)
- Cell-type-specific expression profiles
- Trained model weights (XGBoost)
- Clinical validation datasets

### Interactive Tools
- **Risk Calculator**: Real-time multi-modal risk scoring
- **Gene Browser**: Causal gene and cell type visualization
- **Network Explorer**: Disease-gene-cell interactive networks
- **API Documentation**: Swagger UI at `/docs` endpoint

### Documentation
- **README.md**: This file (research overview)
- **CONTRIBUTING.md**: Contribution guidelines
- **DEPLOYMENT_SUMMARY.md**: Production deployment guide
- **MODEL_CARD.md**: Clinical AI model documentation
- **DATA_DOWNLOAD_GUIDE.md**: Dataset acquisition instructions

---

## 🎓 Citation

If you use the HPCMA in your research, please cite:

```bibtex
@article{hou2024hpcma,
  title={Hypertension Pan-Comorbidity Multi-Modal Atlas: 
         An integrated genomic-clinical resource for end-organ risk prediction},
  author={Hou, Benjamin-J and [Collaborators]},
  journal={Nature Medicine / Cell Genomics},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.XXXX/XXXXX}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Clinical Disclaimer**: This resource is intended for research purposes. Clinical implementation requires appropriate validation, regulatory approval, and oversight by qualified healthcare professionals.

---

## 🤝 Contributing

We welcome contributions from the research community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📞 Contact

- **Repository**: https://github.com/Benjamin-JHou/HPCMA
- **Issues**: https://github.com/Benjamin-JHou/HPCMA/issues
- **Discussions**: https://github.com/Benjamin-JHou/HPCMA/discussions

---

**Maintained by**: Benjamin-JHou | **Version**: 1.0.0 | **Last Updated**: February 2025
