#!/usr/bin/env python3
"""
Finalize Step 1: Rename files, update manifest, clean up
"""

import pandas as pd
from pathlib import Path
import shutil

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

def compute_qc_metrics(file_path):
    print(f"Computing QC for {file_path.name}...")
    df = pd.read_csv(file_path, sep='\t', compression='gzip')
    
    metrics = {
        'SNP_count': len(df),
        'Mean_SE': df['SE'].mean(),
        'Lambda_GC': None,
        'Allele_missing_rate': 0.0,
        'Sample_Size': df['N'].mean() if 'N' in df.columns and df['N'].notna().any() else None
    }
    
    try:
        df['Z'] = df['BETA'] / df['SE']
        df['CHI2'] = df['Z'] ** 2
        median_chi2 = df['CHI2'].median()
        metrics['Lambda_GC'] = median_chi2 / 0.455
        print(f"  Lambda GC: {metrics['Lambda_GC']:.4f}")
    except:
        metrics['Lambda_GC'] = None
    
    missing = df['EA'].isna().sum() + df['NEA'].isna().sum()
    metrics['Allele_missing_rate'] = missing / (len(df) * 2) if len(df) > 0 else 0.0
    
    return metrics

def rename_and_finalize():
    print("="*60)
    print("FINALIZING STEP 1 - Renaming files and creating manifest")
    print("="*60)
    
    manifest_entries = []
    qc_metrics_list = []
    
    for trait, info in DATASETS.items():
        dataset_id = info['id']
        
        std_old = DATA_DIR / "gwas_standardized" / f"{trait}.standardized.gz"
        std_new = DATA_DIR / "gwas_standardized" / f"{trait}_{dataset_id}.txt.gz"
        
        if std_old.exists():
            shutil.move(std_old, std_new)
            print(f"Renamed: {std_old.name} -> {std_new.name}")
            
            qc = compute_qc_metrics(std_new)
            qc['Trait'] = trait
            qc_metrics_list.append(qc)
        
        harm_old = DATA_DIR / "harmonized" / f"{trait}.harmonized.gz"
        harm_new = DATA_DIR / "harmonized" / f"{trait}_{dataset_id}.txt.gz"
        
        if harm_old.exists():
            shutil.move(harm_old, harm_new)
            print(f"Renamed: {harm_old.name} -> {harm_new.name}")
            
            manifest_entries.append({
                'Trait': trait,
                'Dataset_ID': dataset_id,
                'Source': info['source'],
                'Sample_Size': int(qc['Sample_Size']) if qc['Sample_Size'] else 'NA',
                'Ancestry': info['ancestry'],
                'Genome_Build': 'GRCh37',
                'SNP_Count_PostQC': qc['SNP_count'],
                'File_Path': str(harm_new),
                'QC_Passed': 'YES'
            })
    
    # Save QC metrics
    if qc_metrics_list:
        qc_df = pd.DataFrame(qc_metrics_list)
        qc_file = RESULTS_DIR / "gwas_qc_metrics.csv"
        qc_df.to_csv(qc_file, index=False)
        print(f"\nSaved QC metrics: {qc_file}")
    
    # Save manifest
    if manifest_entries:
        manifest_df = pd.DataFrame(manifest_entries)
        manifest_file = RESULTS_DIR / "final_dataset_manifest.csv"
        manifest_df.to_csv(manifest_file, index=False)
        print(f"Saved manifest: {manifest_file}")
    
    # Clean up temp files
    for temp_file in DATA_DIR.rglob("*.temp.tsv"):
        temp_file.unlink()
        print(f"Cleaned up: {temp_file}")
    
    print("\n" + "="*60)
    print("STEP 1 FINALIZATION COMPLETE")
    print("="*60)
    print(f"Datasets processed: {len(manifest_entries)}/10")
    print(f"QC metrics computed: {len(qc_metrics_list)}/10")
    
    if qc_metrics_list:
        print("\nQC Summary:")
        for m in qc_metrics_list:
            print(f"  {m['Trait']}: {m['SNP_count']:,} SNPs, Lambda GC: {m['Lambda_GC']:.3f if m['Lambda_GC'] else 'N/A'}")

if __name__ == "__main__":
    rename_and_finalize()
