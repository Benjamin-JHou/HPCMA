# Paper Materials

## Publication-Ready Figures and Methods

This directory contains materials for the HPCMA manuscript submission.

---

## 📁 Directory Structure

```
paper/
├── figures/           # High-resolution publication figures
├── methods/           # Detailed methods supplement
└── README.md         # This file
```

---

## 🎨 Figures (`figures/`)

### Publication-Ready Figures (300 DPI, CMYK)

| Figure | Description | File |
|--------|-------------|------|
| Fig 1 | Genetic correlation heatmap | `ldsc_rg_heatmap.png` |
| Fig 2 | Multi-layer network graph | `multilayer_network_graph.png` |
| Fig 3 | Causal gene prioritization tiers | `prioritized_gene_tiers.png` |
| Fig 4 | Cell-type mechanism heatmap | `gene_celltype_heatmap.png` |
| Fig 5 | Model performance ROC curves | `model_roc_curves.png` |
| Fig 6 | Risk stratification plot | `risk_stratification_plot.png` |

### Supplementary Figures

| Figure | Description | File |
|--------|-------------|------|
| S1 | Colocalization PPH4 distribution | `coloc_pph4_histogram.png` |
| S2 | MR forest plot | `mr_forest_top_pairs.png` |
| S3 | Shared loci barplot | `shared_loci_barplot.png` |
| S4 | SNP overlap heatmap | `shared_snp_overlap_heatmap.png` |
| S5 | Mechanism axis Sankey | `mechanism_axis_sankey.png` |
| S6 | Gene influence scores | `gene_influence_barplot.png` |
| S7 | SHAP feature importance | `shap_summary.png` |
| S8 | Feature importance barplot | `feature_importance_barplot.png` |
| S9 | Drug target enrichment | `drug_target_enrichment_plot.png` |
| S10 | Clinical translation heatmap | `clinical_translation_heatmap.png` |
| S11 | Mechanism score barplot | `mechanism_score_barplot.png` |
| S12 | Gene tau scores | `gene_tau_barplot.png` |
| S13 | Atlas stability plot | `atlas_stability_plot.png` |

---

## 📝 Methods Supplement (`methods/`)

### Detailed Protocols

1. **GWAS Harmonization Protocol** (Step 1)
   - Dataset QC procedures
   - Genome build alignment
   - Allele frequency harmonization

2. **LD Score Regression** (Step 2)
   - Genetic correlation estimation
   - Heritability analysis
   - Cross-trait meta-analysis

3. **Mendelian Randomization** (Step 3)
   - Instrument selection criteria
   - Sensitivity analyses (MR-Egger, weighted median)
   - Pleiotropy assessment

4. **Colocalization Analysis** (Step 3)
   - Prior specification
   - PPH4 interpretation
   - Fine-mapping validation

5. **Cell-Type Mapping** (Step 4)
   - Single-cell data integration
   - Specificity scoring (Tau)
   - Mechanism axis assignment

6. **Machine Learning** (Step 5)
   - Model architectures
   - Hyperparameter tuning
   - Cross-validation procedures
   - SHAP explainability

7. **Clinical Translation** (Step 6)
   - Risk score interpretation
   - Intervention mapping
   - Bias assessment

---

## 📊 Figure Specifications

### Requirements for Journal Submission

- **Resolution**: 300 DPI minimum
- **Color Mode**: CMYK (for print) + RGB (for web)
- **Formats**: PNG (high-res), PDF (vector), TIFF (alternative)
- **Font**: Arial, minimum 8pt for labels
- **Sizing**: Single column (8.5 cm), Double column (17.5 cm)

### Figure Preparation Checklist

- [ ] 300 DPI resolution confirmed
- [ ] CMYK color space applied
- [ ] Editable source files saved
- [ ] Statistical significance indicated
- [ ] Scale bars included where applicable
- [ ] Sample sizes reported in legends

---

## 🎯 Target Journals

**Primary Targets**:
1. Nature Medicine
2. Cell Genomics
3. Nature Genetics
4. European Heart Journal

**Alternative Targets**:
1. Circulation
2. Hypertension (AHA)
3. PLoS Genetics
4. eLife

---

## 📚 Citation Requirements

All figures and methods must cite:

```bibtex
@article{hou2024hpcma,
  title={Hypertension Pan-Comorbidity Multi-Modal Atlas: 
         An integrated genomic-clinical resource},
  author={Hou, Benjamin-J and [Collaborators]},
  journal={[Target Journal]},
  year={2024}
}
```

---

## 📞 For Reviewers

All data and code underlying these figures are available:
- **Repository**: https://github.com/Benjamin-JHou/HPCMA
- **Atlas Tables**: `../atlas_resource/`
- **Analysis Code**: `../scripts/`

---

**Last Updated**: February 2025 | **Version**: 1.0.0
