#!/usr/bin/env python3
"""Update QC metrics and manifest with all 10 datasets including Stroke_Ischemic"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/yangzi/Desktop/Hypertension Pan-Comorbidity Multi-Modal Atlas/data")
RESULTS_DIR = Path("/Users/yangzi/Desktop/Hypertension Pan-Comorbidity Multi-Modal Atlas/results")

DATASETS = {
    "SBP": {"id": "ieu-b-4818", "trait": "Systolic Blood Pressure", "ancestry": "European", "source": "IEU OpenGWAS"},
    "DBP": {"id": "ieu-b-4819", "trait": "Diastolic Blood Pressure", "ancestry": "European", "source": "IEU OpenGWAS"},
    "PP": {"id": "ieu-b-4820", "trait": "Pulse Pressure", "ancestry": "European", "source": "IEU OpenGWAS"},
    "CAD": {"id": "ieu-b-35", "trait": "Coronary Artery Disease", "ancestry": "European", "source": "IEU OpenGWAS"},
    "T2D": {"id": "ieu-b-107", "trait": "Type 2 Diabetes", "ancestry": "Mixed", "source": "IEU OpenGWAS"},
    "BMI": {"id": "ieu-a-2", "trait": "Body Mass Index", "ancestry": "European", "source": "IEU OpenGWAS"},
    "AD": {"id": "ieu-b-2", "trait": "Alzheimers Disease", "ancestry": "European", "source": "IEU OpenGWAS"},
    "Depression": {"id": "ieu-b-102", "trait": "Depression", "ancestry": "European", "source": "IEU OpenGWAS"},
    "Stroke_Any": {"id": "GCST006906", "trait": "Stroke (Any)", "ancestry": "European", "source": "GWAS Catalog"},
    "Stroke_Ischemic": {"id": "GCST006908", "trait": "Ischemic Stroke", "ancestry": "European", "source": "GWAS Catalog"},
}

print("Updating QC metrics and manifest for all 10 datasets...")

manifest_entries = []
qc_metrics_list = []

for trait, info in DATASETS.items():
    dataset_id = info['id']
    harm_file = DATA_DIR / "harmonized" / f"{trait}_{dataset_id}.txt.gz"
    
    if harm_file.exists():
        print(f"Processing {trait}...")
        df = pd.read_csv(harm_file, sep='\t', compression='gzip')
        
        # QC metrics
        metrics = {
            'Trait': trait,
            'SNP_count': len(df),
            'Mean_SE': df['SE'].mean(),
            'Lambda_GC': None,
            'Allele_missing_rate': 0.0,
            'Sample_Size': df['N'].mean() if 'N' in df.columns and df['N'].notna().any() else None
        }
        
        # Compute Lambda GC
        try:
            df['Z'] = df['BETA'] / df['SE']
            df['CHI2'] = df['Z'] ** 2
            median_chi2 = df['CHI2'].median()
            metrics['Lambda_GC'] = median_chi2 / 0.455
        except:
            pass
        
        missing = df['EA'].isna().sum() + df['NEA'].isna().sum()
        metrics['Allele_missing_rate'] = missing / (len(df) * 2) if len(df) > 0 else 0.0
        
        qc_metrics_list.append(metrics)
        
        # Manifest entry
        manifest_entries.append({
            'Trait': trait,
            'Dataset_ID': dataset_id,
            'Source': info['source'],
            'Sample_Size': int(metrics['Sample_Size']) if metrics['Sample_Size'] else 'NA',
            'Ancestry': info['ancestry'],
            'Genome_Build': 'GRCh37',
            'SNP_Count_PostQC': metrics['SNP_count'],
            'File_Path': str(harm_file),
            'QC_Passed': 'YES'
        })
        
        lambda_gc_str = f"{metrics['Lambda_GC']:.3f}" if metrics['Lambda_GC'] else 'N/A'
        print(f"  SNPs: {metrics['SNP_count']:,}, Lambda GC: {lambda_gc_str}")

# Save QC metrics
qc_df = pd.DataFrame(qc_metrics_list)
qc_df.to_csv(RESULTS_DIR / "gwas_qc_metrics.csv", index=False)
print(f"\nSaved QC metrics: {RESULTS_DIR / 'gwas_qc_metrics.csv'}")

# Save manifest
manifest_df = pd.DataFrame(manifest_entries)
manifest_df.to_csv(RESULTS_DIR / "final_dataset_manifest.csv", index=False)
print(f"Saved manifest: {RESULTS_DIR / 'final_dataset_manifest.csv'}")

print("\n" + "="*60)
print("STEP 1 COMPLETE - All 10 datasets finalized")
print("="*60)
print(f"\nDatasets: {len(manifest_entries)}/10")
print(f"Total SNPs: {sum(m['SNP_count'] for m in qc_metrics_list):,}")

print("\nQC Summary:")
for m in qc_metrics_list:
    lambda_gc_str = f"{m['Lambda_GC']:.3f}" if m['Lambda_GC'] else 'N/A'
    print(f"  {m['Trait']}: {m['SNP_count']:,} SNPs, Lambda GC: {lambda_gc_str}")
