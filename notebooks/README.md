# HPCMA Notebooks

Interactive tutorials and examples for the Hypertension Pan-Comorbidity Multi-Modal Atlas.

## 📚 Available Notebooks

### 1. Atlas Query Examples (`atlas_query_examples.ipynb`)
Comprehensive tutorial covering:
- Gene-centric queries (ACE, NOS3, UMOD)
- Disease-centric queries (CAD, CKD, Stroke)
- Patient risk assessment with MMRS
- Cross-disease gene analysis
- Cell type enrichment visualization
- Network exploration

**Run this first** to understand how to query the atlas database.

## 🚀 Quick Start

```python
# Load atlas data
import pandas as pd

# Master table: Gene → Disease → Cell Type
master_df = pd.read_csv('../atlas_resource/hypertension_atlas_master_table.csv')

# Query example: Find all information for ACE gene
ace_data = master_df[master_df['gene'] == 'ACE']
print(ace_data[['disease', 'cell_type', 'mechanism_axis', 'clinical_intervention']])
```

## 📖 Documentation

- **Full Guide**: See `../ATLAS_USAGE_GUIDE.md`
- **Data Dictionary**: See `../atlas_data_dictionary.csv`
- **API Docs**: Run `python -m src.inference.api_server` and visit `http://localhost:8000/docs`

## 🎯 Common Queries

### Query by Gene
```python
gene = 'NOS3'
gene_info = master_df[master_df['gene'] == gene]
```

### Query by Disease
```python
disease = 'CKD'
disease_genes = master_df[master_df['disease'] == disease]
```

### Query by Cell Type
```python
cell_df = pd.read_csv('../atlas_resource/gene_disease_celltype_annotation.csv')
endothelial = cell_df[cell_df['cell_type'] == 'Endothelial']
```

## 🎓 Citation

See main repository README.md for citation information.

---

**Maintained by**: Benjamin-JHou | **Repository**: https://github.com/Benjamin-JHou/HPCMA
