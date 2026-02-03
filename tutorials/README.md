# HPCMA Tutorials

## Step-by-Step Interactive Tutorials

This directory contains Jupyter notebooks for each step of the HPCMA pipeline.

---

## 📚 Tutorial Series

### Beginner Track

| Tutorial | Topic | Description |
|----------|-------|-------------|
| **01** | Atlas Query Examples | Learn how to query the public atlas resource |

### Advanced Track (7 Steps)

| Tutorial | Step | Topic | Description |
|----------|------|-------|-------------|
| **02** | Step 1 | GWAS Harmonization | Dataset QC and harmonization procedures |
| **03** | Step 2 | Genetic Architecture | LD Score Regression and cross-trait analysis |
| **04** | Step 3 | Causal Gene Prioritization | MR and colocalization methods |
| **05** | Step 4 | Cell-Type Mapping | Single-cell integration and specificity |
| **06** | Step 5 | Multi-Modal Prediction | ML model training and ensemble |
| **07** | Step 6-7 | Atlas Integration & Validation | Final assembly and validation |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

### Launch Jupyter

```bash
cd tutorials
jupyter notebook
```

### Run Tutorial 1 (Recommended First)

1. Open `01_atlas_query_examples.ipynb`
2. Execute cells sequentially
3. Explore gene-disease-cell mappings
4. Practice with interactive examples

---

## 📊 Learning Path

### For Researchers

**Path**: 01 → 04 → 07
- Learn atlas querying
- Understand causal gene prioritization
- Master validation protocols

### For Methodologists

**Path**: 01 → 02 → 03 → 04 → 05 → 06 → 07
- Complete pipeline walkthrough
- Understand each methodological step
- Learn implementation details

### For Clinicians

**Path**: 01 → 05 → 07
- Query the atlas
- Understand risk prediction
- Learn clinical translation

---

## 🎯 Tutorial Objectives

### Tutorial 01: Atlas Query Examples
- Query genes and retrieve disease associations
- Explore cell-type-specific mechanisms
- Calculate patient risk scores (MMRS)
- Navigate the gene-disease-cell network

### Tutorial 02: GWAS Harmonization
- Load and QC GWAS summary statistics
- Harmonize genome builds
- Align allele frequencies
- Filter and validate datasets

### Tutorial 03: Genetic Architecture
- Calculate genetic correlations
- Interpret LD Score Regression results
- Visualize cross-trait relationships
- Identify shared loci

### Tutorial 04: Causal Gene Prioritization
- Select genetic instruments
- Run Mendelian Randomization
- Assess colocalization evidence
- Integrate multiple evidence sources

### Tutorial 05: Cell-Type Mapping
- Access single-cell atlases
- Calculate specificity scores (Tau)
- Map genes to cell types
- Assign mechanism axes

### Tutorial 06: Multi-Modal Prediction
- Engineer multi-modal features
- Train XGBoost/RF models
- Perform cross-validation
- Generate SHAP explanations

### Tutorial 07: Integration & Validation
- Build multi-layer networks
- Create clinical translation maps
- Validate predictions
- Assess bias and fairness

---

## 💡 Tips

### Interactive Learning
- Modify parameters in tutorial cells
- Try different genes (ACE, NOS3, UMOD)
- Experiment with disease queries
- Generate custom visualizations

### Common Patterns
```python
# Pattern 1: Gene query
import pandas as pd
master = pd.read_csv('../atlas_resource/hypertension_atlas_master_table.csv')
gene_data = master[master['Gene'] == 'ACE']

# Pattern 2: Disease query
disease_genes = master[master['Disease'] == 'CAD']

# Pattern 3: Cell type query
cell_data = pd.read_csv('../atlas_resource/gene_disease_celltype_annotation.csv')
cell_specific = cell_data[cell_data['CellType'] == 'Endothelial']
```

---

## 📖 Documentation

- **Full Guide**: See `../ATLAS_USAGE_GUIDE.md`
- **Methods**: See `../paper/methods/README.md`
- **Data Dictionary**: See `../atlas_data_dictionary.csv`

---

## 🆘 Troubleshooting

### Issue: File not found
**Solution**: Ensure you're running notebooks from the `tutorials/` directory

### Issue: Import errors
**Solution**: Install required packages:
```bash
pip install -r ../requirements.txt
```

### Issue: Memory error
**Solution**: Restart kernel and run cells one at a time

---

## 🤝 Contributing

Have ideas for new tutorials? Open an issue on GitHub!

---

**Last Updated**: February 2025 | **Version**: 1.0.0
