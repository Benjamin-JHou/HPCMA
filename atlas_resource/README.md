# HPCMA Atlas Resource

## Public Biological Atlas Tables

This directory contains the core public resource tables from the Hypertension Pan-Comorbidity Multi-Modal Atlas (HPCMA). These tables are freely available for research use.

---

## 📊 Available Tables

### Core Atlas Tables

| Table | Description | Records | Key Use |
|-------|-------------|---------|---------|
| `hypertension_atlas_master_table.csv` | Central gene-disease-cell mapping | ~500 | Primary query table |
| `prioritized_causal_genes.csv` | Tier 1-3 causal genes with evidence | 50+ | Causal gene discovery |
| `gene_disease_celltype_annotation.csv` | Cell-type-specific disease mechanisms | ~300 | Cell type exploration |
| `clinical_translation_table.csv` | Gene → Intervention mapping | 25+ | Clinical applications |
| `mechanism_axis_clusters.csv` | Biological pathway groupings | 3 axes | Mechanism exploration |

### Supporting Evidence Tables

| Table | Description | Records | Key Use |
|-------|-------------|---------|---------|
| `ldsc_genetic_correlation_matrix.csv` | Genetic correlations (rg) | 55 pairs | Shared architecture |
| `coloc_results.csv` | Colocalization evidence | 13 loci | Shared causal variants |
| `multilayer_network_edges.csv` | Disease-gene-cell network | 17 edges | Network analysis |

---

## 🔬 Data Dictionary

Complete column definitions: See `../atlas_data_dictionary.csv`

Quick reference:
- **Gene**: HGNC gene symbol
- **Disease**: CAD, Stroke, CKD, T2D, Depression, AD
- **Tier**: 1 (high confidence), 2 (moderate), 3 (low)
- **MR_Beta**: Mendelian randomization effect size
- **PPH4**: Colocalization posterior probability
- **CellType**: Disease-relevant cell population
- **Mechanism_Axis**: Vascular Tone, Renal Salt, Cardiac Natriuretic

---

## 🚀 Quick Start

### Query Examples

```python
import pandas as pd

# Load master atlas
master = pd.read_csv('hypertension_atlas_master_table.csv')

# Query gene
ace_data = master[master['Gene'] == 'ACE']

# Query disease
cad_genes = master[master['Disease'] == 'CAD']

# Query mechanism axis
vascular = master[master['Mechanism_Axis'] == 'Vascular Tone Regulation']
```

### Atlas Statistics

- **Total Genes**: 50+ prioritized causal genes
- **Tier 1 Genes**: 7 (ACE, AGT, EDN1, NOS3, NPPA, SHROOM3, UMOD)
- **Diseases**: 6 comorbidities
- **Cell Types**: 15+ disease-relevant populations
- **Mechanism Axes**: 3 major pathways

---

## 📖 Citation

When using these tables:

```bibtex
@article{hou2024hpcma,
  title={Hypertension Pan-Comorbidity Multi-Modal Atlas},
  author={Hou, Benjamin-J and [Collaborators]},
  journal={Nature Medicine / Cell Genomics},
  year={2024}
}
```

---

## 🔗 Related Documentation

- **Full Usage Guide**: See `../ATLAS_USAGE_GUIDE.md`
- **Interactive Examples**: See `../notebooks/atlas_query_examples.ipynb`
- **Validation Notes**: See `../VALIDATION_WHITEPAPER.md`

---

**License**: MIT | **Last Updated**: February 2025
