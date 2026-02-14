# HPCMA Atlas Usage Guide

## 🗺️ Public Biological Atlas Resource

Welcome to the **Hypertension Pan-Comorbidity Multi-Modal Atlas (HPCMA)**! This guide will help you navigate and query our comprehensive database of hypertension-mediated multi-organ disease mechanisms.

---

## 📚 Table of Contents

1. [Atlas Overview](#atlas-overview)
2. [Querying by Gene](#querying-by-gene)
3. [Querying by Disease](#querying-by-disease)
4. [Querying Mechanism Axes](#querying-mechanism-axes)
5. [Querying Intervention Mappings](#querying-intervention-mappings)
6. [Advanced Queries](#advanced-queries)
7. [Data Dictionary](#data-dictionary)

---

## Atlas Overview

### What is HPCMA?

The Hypertension Pan-Comorbidity Multi-Modal Atlas is an integrated biological database connecting:

- 🧬 **Genes** (causal variants from GWAS/MR/Coloc)
- 🏥 **Diseases** (6 comorbidities: CAD, Stroke, CKD, T2D, Depression, AD)
- 🔬 **Cell Types** (disease-relevant cell populations)
- 💊 **Interventions** (clinical recommendations)

### Core Atlas Tables

| Table | Purpose | Records |
|-------|---------|---------|
| `hypertension_atlas_master_table.csv` | Central gene-disease-cell mapping | ~500 entries |
| `prioritized_causal_genes.csv` | Tier 1-3 causal genes with evidence | 50+ genes |
| `gene_disease_celltype_annotation.csv` | Cell-type-specific disease mechanisms | ~300 entries |
| `clinical_translation_table.csv` | Gene → Intervention mapping | 25+ interventions |
| `mechanism_axis_clusters.csv` | Biological pathway groupings | 3 major axes |
| `multilayer_network_edges.csv` | Network relationships | 17 edges |

### Atlas Statistics

- **Total Genes**: 50+ prioritized causal genes
- **Tier 1 Genes**: 7 high-confidence causal genes
- **Diseases**: 6 comorbidities mapped
- **Cell Types**: 15+ disease-relevant cell types
- **Mechanism Axes**: 3 major biological pathways
- **Interventions**: 25+ clinical recommendations

---

## Querying by Gene

### Basic Gene Query: Input → Output

**Input**: Gene Symbol (e.g., `ACE`, `NOS3`, `UMOD`)

**Output**: Diseases + Cell Types + Mechanisms + Interventions

### Example Query 1: Find All Information for Gene ACE

```python
import pandas as pd

# Load the master atlas table
master_df = pd.read_csv('atlas_resource/hypertension_atlas_master_table.csv')

# Query for ACE gene
ace_info = master_df[master_df['gene'] == 'ACE']

# Display results
print("=== ACE Gene Atlas Information ===")
print(f"Diseases associated: {ace_info['disease'].unique()}")
print(f"Cell types: {ace_info['cell_type'].unique()}")
print(f"Mechanism axis: {ace_info['mechanism_axis'].iloc[0]}")
print(f"Clinical intervention: {ace_info['clinical_intervention'].iloc[0]}")
```

**Expected Output:**
```
=== ACE Gene Atlas Information ===
Diseases associated: ['CAD' 'Stroke' 'CKD']
Cell types: ['Endothelial' 'Mesangial']
Mechanism axis: Vascular Tone Regulation
Clinical intervention: ACE inhibitors (e.g., Lisinopril)
```

### Example Query 2: Get Detailed Gene Evidence

```python
# Load prioritized genes table
genes_df = pd.read_csv('atlas_resource/prioritized_causal_genes.csv')

# Query specific gene
gene_symbol = 'NOS3'  # Change to query different genes
gene_details = genes_df[genes_df['gene'] == gene_symbol].iloc[0]

print(f"=== {gene_symbol} Evidence Summary ===")
print(f"Priority Tier: {gene_details['Tier']}")
print(f"MR Support: {gene_details['MR_Support']}")
print(f"Coloc Support: {gene_details['Coloc_Support']}")
print(f"eQTL Support: {gene_details['eQTL_Support']}")
print(f"Priority Score: {gene_details['priority_score']:.2f}")
print(f"Associated Traits: {gene_details['Associated_Traits']}")
```

### Example Query 3: Find Genes by Cell Type

```python
# Load cell-type annotation table
cell_df = pd.read_csv('atlas_resource/gene_disease_celltype_annotation.csv')

# Find all genes expressed in endothelial cells
endothelial_genes = cell_df[
    (cell_df['cell_type'] == 'Endothelial') & 
    (cell_df['is_disease_relevant'] == True)
]

print(f"Disease-relevant genes in endothelial cells:")
print(endothelial_genes['gene'].unique())
print(f"\nMechanisms: {endothelial_genes['Mechanism'].unique()}")
```

---

## Querying by Disease

### Disease → Gene → Cell Type Chain

**Input**: Disease Name (e.g., `CAD`, `CKD`, `Stroke`)

**Output**: All causal genes + their cell types + mechanisms

### Example Query 4: Map Disease to Mechanism Axis

```python
# Load mechanism clusters
mech_df = pd.read_csv('atlas_resource/mechanism_axis_clusters.csv')

# Query by disease (encoded in mechanism axis description)
disease = 'CAD'
cad_mechanisms = mech_df[mech_df['Axis_Description'].str.contains(disease, case=False, na=False)]

print(f"=== {disease} Mechanism Axis ===")
print(f"Axis: {cad_mechanisms['mechanism_axis'].iloc[0]}")
print(f"Description: {cad_mechanisms['Axis_Description'].iloc[0]}")
print(f"Key Genes: {', '.join(cad_mechanisms['gene'].unique())}")
print(f"Cell Types: {', '.join(cad_mechanisms['cell_type'].unique())}")
```

### Example Query 5: Get All Genes for a Specific Disease

```python
# Load master table
master_df = pd.read_csv('atlas_resource/hypertension_atlas_master_table.csv')

disease = 'CKD'
disease_genes = master_df[master_df['disease'] == disease]

print(f"=== {disease}: Causal Genes & Cell Types ===")
for _, row in disease_genes.iterrows():
    print(f"Gene: {row['gene']}")
    print(f"  Cell Type: {row['cell_type']}")
    print(f"  Tissue: {row['Tissue']}")
    print(f"  Mechanism: {row['mechanism_axis']}")
    print(f"  MR Effect: {row['mr_beta']:.3f}")
    print(f"  Priority Score: {row['priority_score']:.1f}")
    print()
```

### Example Query 6: Cross-Disease Gene Analysis

```python
# Load cross-disease influence table
cross_df = pd.read_csv('results/cross_disease_gene_influence_score.csv')

# Find genes that influence multiple diseases
pleiotropic_genes = cross_df[cross_df['n_diseases_involved'] >= 3]

print("=== Pleiotropic Genes (3+ Diseases) ===")
for _, gene in pleiotropic_genes.iterrows():
    print(f"{gene['gene']}: {gene['n_diseases_involved']} diseases")
    print(f"  Top Cell Type: {gene['top_cell_type']}")
    print(f"  Mechanism Axis: {gene['mechanism_axis']}")
    print(f"  Influence Score: {gene['total_influence_score']:.2f}")
    print()
```

---

## Querying Mechanism Axes

### The Three Major Mechanism Axes

1. **Vascular Tone Regulation** (ACE, AGT, EDN1, NOS3)
2. **Renal Salt Handling** (SHROOM3, UMOD)
3. **Cardiac Natriuretic Signaling** (NPPA)

### Example Query 7: Explore Mechanism Axis

```python
mech_df = pd.read_csv('atlas_resource/mechanism_axis_clusters.csv')

axis_name = 'Vascular Tone Regulation'
axis_genes = mech_df[mech_df['mechanism_axis'] == axis_name]

print(f"=== {axis_name} Mechanism Axis ===")
print(f"Biological Mechanism: {axis_genes['Biological_Mechanism'].iloc[0]}")
print(f"\nGenes in this axis:")
for gene in axis_genes['gene'].unique():
    gene_data = axis_genes[axis_genes['gene'] == gene].iloc[0]
    print(f"  • {gene}: {gene_data['cell_type']} (Score: {gene_data['mechanism_score']:.2f})")
```

### Example Query 8: Find Genes by Mechanism Score

```python
# Find high-confidence mechanism genes
high_confidence = mech_df[mech_df['mechanism_score'] > 0.8]

print("=== High Confidence Mechanism Genes (Score > 0.8) ===")
for axis in high_confidence['mechanism_axis'].unique():
    axis_high = high_confidence[high_confidence['mechanism_axis'] == axis]
    print(f"\n{axis}:")
    for _, row in axis_high.iterrows():
        print(f"  {row['gene']} ({row['cell_type']}): {row['mechanism_score']:.2f}")
```

---

## Querying Intervention Mappings

### Gene → Clinical Intervention Mapping

**Input**: Gene Symbol
**Output**: Evidence-based clinical interventions

### Example Query 9: Get Interventions for Gene

```python
# Load clinical translation table
clinical_df = pd.read_csv('atlas_resource/clinical_translation_table.csv')

gene = 'ACE'
interventions = clinical_df[clinical_df['gene'] == gene]

print(f"=== Clinical Interventions for {gene} ===")
for _, intervention in interventions.iterrows():
    print(f"\nIntervention: {intervention['clinical_intervention']}")
    print(f"  Risk Factor: {intervention['risk_factor']}")
    print(f"  Pathway: {intervention['Pathway']}")
    print(f"  Evidence Level: {intervention['evidence_level']}")
    print(f"  BP Effect: {intervention['BP_Effect']}")
    print(f"  Comorbidity Benefit: {intervention['Comorbidity_Benefit']}")
    print(f"  Monitoring: {intervention['Monitoring']}")
```

### Example Query 10: Find All Genes with Drug Targets

```python
# Load drug target enrichment table
drug_df = pd.read_csv('results/drug_target_enrichment_results.csv')

# Filter significant drug targets
approved_targets = drug_df[
    (drug_df['Significant'] == True) & 
    (drug_df['Approved_Drug_Targets'] > 0)
]

print("=== Genes with Approved Drug Targets ===")
for _, target in approved_targets.iterrows():
    print(f"\nGene Set: {target['gene_set']}")
    print(f"  Approved Drug Targets: {target['Approved_Drug_Targets']}")
    print(f"  Enrichment Ratio: {target['Enrichment_Ratio']:.2f}")
    print(f"  P-value: {target['P_Value']:.2e}")
    print(f"  Interpretation: {target['Interpretation']}")
```

---

## Advanced Queries

### Query 11: Multi-Modal Patient Risk Query

**Input**: Patient PRS scores
**Output**: Multi-Modal Risk Score (MMRS)

```python
# Load risk score table
risk_df = pd.read_csv('results/multimodal_risk_score.csv')

# Example: Query specific patient
patient_id = 'Sample_001'
patient_risks = risk_df[risk_df['Sample_ID'] == patient_id].iloc[0]

print(f"=== Multi-Modal Risk Profile: {patient_id} ===")
diseases = ['CAD', 'Stroke', 'CKD', 'T2D', 'Depression', 'AD']
for disease in diseases:
    risk = patient_risks[disease]
    risk_level = 'LOW' if risk < 0.15 else 'MODERATE' if risk < 0.3 else 'HIGH' if risk < 0.45 else 'VERY HIGH'
    print(f"{disease}: {risk:.3f} ({risk_level})")

# Calculate MMRS
mmrs = risk_df[risk_df['Sample_ID'] == patient_id][diseases].mean(axis=1).iloc[0]
print(f"\nMMRS (Composite): {mmrs:.3f}")
```

### Query 12: Network-Based Gene Discovery

```python
# Load network edges
network_df = pd.read_csv('atlas_resource/multilayer_network_edges.csv')

# Find hub genes (genes connected to multiple diseases)
gene_connections = network_df[network_df['Layer'] == 'Gene_Disease'].groupby('Source').size()
hub_genes = gene_connections[gene_connections >= 3]

print("=== Hub Genes (Connected to 3+ Diseases) ===")
for gene, n_connections in hub_genes.items():
    print(f"{gene}: {n_connections} disease connections")
    
# Get details for hub genes
for gene in hub_genes.index[:5]:  # Top 5
    connections = network_df[
        (network_df['Layer'] == 'Gene_Disease') & 
        (network_df['Source'] == gene)
    ]
    print(f"\n{gene} connections:")
    for _, conn in connections.iterrows():
        print(f"  → {conn['Target']} (Evidence: {conn['Evidence_Type']})")
```

### Query 13: eQTL-Supported Genes

```python
# Load eQTL results
eqtl_df = pd.read_csv('results/eqtl_supported_genes.csv')

# Find genes with strong eQTL support
strong_eqtl = eqtl_df[eqtl_df['eQTL_Support'] == True]

print("=== Genes with eQTL Support (Expression Quantitative Trait Loci) ===")
for _, gene in strong_eqtl.iterrows():
    print(f"\n{gene['gene']}:")
    print(f"  Top SNP: {gene['Top_SNP']}")
    print(f"  eQTL P-value: {gene['eQTL_P']:.2e}")
    print(f"  Tissues: {gene['eQTL_Tissues']}")
    print(f"  GTEx v8: {gene['GTEx_v8']}")
```

---

## Data Dictionary

For complete column definitions, see `atlas_data_dictionary.csv`

### Quick Reference: Key Columns

| Column | Table(s) | Description |
|--------|----------|-------------|
| `gene` | All | HGNC gene symbol |
| `disease` | Master | Target comorbidity (CAD/Stroke/CKD/T2D/Depression/AD) |
| `Tier` | Prioritized | Evidence tier (1=high, 2=medium, 3=low) |
| `mr_beta` | Master | Mendelian randomization effect size |
| `pph4` | Master | Colocalization posterior probability |
| `cell_type` | Multiple | Disease-relevant cell type |
| `mechanism_axis` | Multiple | Biological pathway category |
| `clinical_intervention` | Clinical | Recommended intervention |
| `priority_score` | Multiple | Composite evidence score |

---

## 🎓 Citation

When using HPCMA data in your research:

```bibtex
@article{hou2024hpcma,
  title={Hypertension Pan-Comorbidity Multi-Modal Atlas: 
         An integrated genomic-clinical resource for end-organ risk prediction},
  author={Hou, Benjamin-J and [Collaborators]},
  journal={Nature Medicine / Cell Genomics},
  year={2024},
  doi={10.XXXX/XXXXX}
}
```

---

## 📞 Support

- **Issues**: https://github.com/Benjamin-JHou/HPCMA/issues
- **Examples**: See `atlas_query_examples.ipynb` for interactive tutorials
- **Data Dictionary**: See `atlas_data_dictionary.csv` for complete schema

---

**Last Updated**: February 2025 | **Version**: 1.0.0
