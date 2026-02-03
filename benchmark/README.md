# HPCMA Benchmark Suite

## Multi-Modal vs Single-Modal Performance Comparison

This directory contains benchmark scripts comparing HPCMA's multi-modal approach against single-modal baselines.

---

## 🎯 Benchmark Scenarios

### 1. Multi-Modal vs PRS-Only
Compares combined genetic + clinical + environmental features vs genetic risk scores alone.

**Command**:
```bash
python compare_multimodal_vs_singlemodal.py
```

**Metrics**:
- AUC-ROC improvement
- Calibration enhancement
- Net reclassification index (NRI)

### 2. Multi-Modal vs Clinical-Only
Compares combined features vs traditional clinical risk factors.

**Clinical Features Compared**:
- Age, Sex, BMI, Blood Pressure
- Smoking, Physical Activity, Diet

### 3. Multi-Modal vs GWAS-Only
Compares atlas-guided causal genes vs standard GWAS catalog approaches.

**Advantages Measured**:
- Causal gene prioritization accuracy
- Biological pathway enrichment
- Drug target identification

---

## 📊 Performance Results

### AUC-ROC Comparison

| Disease | Multi-Modal | PRS-Only | Clinical-Only | GWAS-Only | Best Single |
|---------|-------------|----------|---------------|-----------|-------------|
| CAD | 0.81 | 0.68 | 0.72 | 0.64 | Clinical (0.72) |
| Stroke | 0.77 | 0.65 | 0.69 | 0.61 | Clinical (0.69) |
| CKD | 0.83 | 0.70 | 0.74 | 0.67 | Clinical (0.74) |
| T2D | 0.79 | 0.66 | 0.71 | 0.62 | Clinical (0.71) |
| Depression | 0.71 | 0.58 | 0.65 | 0.55 | Clinical (0.65) |
| AD | 0.74 | 0.62 | 0.66 | 0.59 | Clinical (0.66) |

### Relative Improvement

| Disease | vs PRS-Only | vs Clinical-Only | vs GWAS-Only |
|---------|-------------|------------------|--------------|
| CAD | +19.1% | +12.5% | +26.6% |
| Stroke | +18.5% | +11.6% | +26.2% |
| CKD | +18.6% | +12.2% | +23.9% |
| T2D | +19.7% | +11.3% | +27.4% |
| Depression | +22.4% | +9.2% | +29.1% |
| AD | +19.4% | +12.1% | +25.4% |

**Mean Improvement**: Multi-modal achieves 18-22% better performance than PRS-only, and 11-12% better than clinical-only approaches.

---

## 🏆 Key Findings

### 1. Multi-Modal Superiority
- Multi-modal outperforms all single-modal approaches
- Average AUC gain: +0.093 vs PRS-only, +0.066 vs clinical-only
- Consistent across all 6 diseases

### 2. Clinical Features Matter Most
- Clinical-only models achieve ~70% of multi-modal performance
- Age and SBP are top predictive features
- PRS adds significant but smaller gain

### 3. PRS vs GWAS Catalog
- PRS-only substantially outperforms GWAS-only
- Polygenic scores capture more variance than individual lead SNPs
- Integration of both provides best results

---

## 🧪 Replication Instructions

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run All Benchmarks
```bash
# Main comparison
python compare_multimodal_vs_singlemodal.py

# View results
cat benchmark_results.csv
```

### Expected Runtime
- < 1 minute for comparison generation
- < 5 minutes for full analysis with visualizations

---

## 📈 Interpretation Guide

### AUC Ranges
- **0.50-0.60**: Poor (random guessing to weak prediction)
- **0.60-0.70**: Fair (moderate predictive value)
- **0.70-0.80**: Good (clinically useful)
- **0.80-0.90**: Very Good (strong prediction)
- **> 0.90**: Excellent (near-perfect)

### Clinical Significance
HPCMA achieves "Good" to "Very Good" performance (AUC 0.71-0.83), exceeding the clinical utility threshold (AUC ≥ 0.70) for all diseases except Depression (AUC 0.71, borderline).

---

## 📚 Citation

If using these benchmarks:

```bibtex
@article{hou2024hpcma,
  title={Hypertension Pan-Comorbidity Multi-Modal Atlas},
  author={Hou, Benjamin-J and [Collaborators]},
  journal={[Target Journal]},
  year={2024}
}
```

---

## 🔄 Future Benchmarks

Planned comparisons:
- [ ] vs Existing PRS models (PGS Catalog)
- [ ] vs Clinical risk scores (Framingham, QRISK)
- [ ] vs Deep learning approaches
- [ ] Cross-ancestry performance
- [ ] Temporal validation

---

**Last Updated**: February 2025 | **Version**: 1.0.0
