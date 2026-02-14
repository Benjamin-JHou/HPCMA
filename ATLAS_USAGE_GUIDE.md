# HPCMA Atlas Usage Guide

## Overview

This guide explains how to use the public HPCMA atlas resource tables for reproducible research.

Canonical tables are in `atlas_resource/` and use `lower_snake_case` field names.

## Core Tables

- `hypertension_atlas_master_table.csv`
- `prioritized_causal_genes.csv`
- `gene_disease_celltype_annotation.csv`
- `clinical_translation_table.csv`
- `mechanism_axis_clusters.csv`
- `multilayer_network_edges.csv`
- `ldsc_genetic_correlation_matrix.csv`
- `coloc_results.csv`

## Basic Query Examples

```python
import pandas as pd

master = pd.read_csv('atlas_resource/hypertension_atlas_master_table.csv')

# gene-level query
ace = master[master['gene'] == 'ACE']

# disease-level query
cad = master[master['disease'] == 'CAD']

# mechanism-axis query
vascular = master[master['mechanism_axis'].str.contains('vascular', case=False, na=False)]
```

## Gene Evidence Query

```python
genes = pd.read_csv('atlas_resource/prioritized_causal_genes.csv')
row = genes[genes['gene'] == 'NOS3'].iloc[0]

print(row['tier'], row['priority_score'], row['mr_support'], row['coloc_support'])
```

## Cell-Type Annotation Query

```python
cell = pd.read_csv('atlas_resource/gene_disease_celltype_annotation.csv')
endo = cell[(cell['cell_type'] == 'Endothelial') & (cell['is_disease_relevant'] == True)]
print(endo['gene'].unique())
```

## Mechanism Axis Query

```python
axis = pd.read_csv('atlas_resource/mechanism_axis_clusters.csv')
subset = axis[axis['mechanism_axis'].str.contains('vascular', case=False, na=False)]
print(subset[['gene', 'cell_type', 'mechanism_score']].head())
```

## Clinical Translation Table Usage

`clinical_translation_table.csv` is provided as a literature-linked research mapping table. It should be interpreted as hypothesis-supporting evidence context, not direct care guidance.

```python
clinical = pd.read_csv('atlas_resource/clinical_translation_table.csv')
print(clinical[clinical['gene'] == 'ACE'][['risk_factor', 'clinical_intervention', 'evidence_level']])
```

## Database Query Contract

Build and query the SQLite release artifact:

```bash
python3 database/build_db.py
sqlite3 release/hpcma_atlas.sqlite ".read database/queries/query_gene_profile.sql"
```

Regression contract is enforced by `tests/test_query_regression.py`.

## Reproducibility Workflow

```bash
python -m pytest tests -q
bash reproducibility/run_all.sh
```

## Scope Reminder

HPCMA is released as a reproducible research resource. Supplementary API artifacts are demonstration-only and out of clinical deployment scope.
