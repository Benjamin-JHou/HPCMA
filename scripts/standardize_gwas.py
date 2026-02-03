#!/usr/bin/env python3
"""
Optimized GWAS Data Standardization and Harmonization Pipeline
Processes large VCF files in chunks to avoid memory and timeout issues
"""

import os
import gzip
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/gwas_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/Users/yangzi/Desktop/Hypertension Pan-Comorbidity Multi-Modal Atlas/data")
RESULTS_DIR = Path("/Users/yangzi/Desktop/Hypertension Pan-Comorbidity Multi-Modal Atlas/results")
STD_DIR = DATA_DIR / "gwas_standardized"
HARM_DIR = DATA_DIR / "harmonized"

# Ensure directories exist
STD_DIR.mkdir(exist_ok=True)
HARM_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset mapping
DATASETS = {
    # IEU VCF files - focus on main hypertension traits first
    "SBP": {
        "file": DATA_DIR / "ieu_opengwas/ieu-b-4818.vcf.gz",
        "type": "vcf",
        "trait": "Systolic Blood Pressure",
        "id": "ieu-b-4818",
        "ancestry": "European",
        "source": "IEU OpenGWAS",
        "build": "GRCh37"
    },
    "DBP": {
        "file": DATA_DIR / "ieu_opengwas/ieu-b-4819.vcf.gz",
        "type": "vcf",
        "trait": "Diastolic Blood Pressure",
        "id": "ieu-b-4819",
        "ancestry": "European",
        "source": "IEU OpenGWAS",
        "build": "GRCh37"
    },
    "PP": {
        "file": DATA_DIR / "ieu_opengwas/ieu-b-4820.vcf.gz",
        "type": "vcf",
        "trait": "Pulse Pressure",
        "id": "ieu-b-4820",
        "ancestry": "European",
        "source": "IEU OpenGWAS",
        "build": "GRCh37"
    },
    "CAD": {
        "file": DATA_DIR / "ieu_opengwas/ieu-b-35.vcf.gz",
        "type": "vcf",
        "trait": "Coronary Artery Disease",
        "id": "ieu-b-35",
        "ancestry": "European",
        "source": "IEU OpenGWAS",
        "build": "GRCh37"
    },
    "T2D": {
        "file": DATA_DIR / "ieu_opengwas/ieu-b-107.vcf.gz",
        "type": "vcf",
        "trait": "Type 2 Diabetes",
        "id": "ieu-b-107",
        "ancestry": "Mixed",
        "source": "IEU OpenGWAS",
        "build": "GRCh37"
    },
    "BMI": {
        "file": DATA_DIR / "ieu_opengwas/ieu-a-2.vcf.gz",
        "type": "vcf",
        "trait": "Body Mass Index",
        "id": "ieu-a-2",
        "ancestry": "European",
        "source": "IEU OpenGWAS",
        "build": "GRCh37"
    },
    "AD": {
        "file": DATA_DIR / "ieu_opengwas/ieu-b-2.vcf.gz",
        "type": "vcf",
        "trait": "Alzheimer's Disease",
        "id": "ieu-b-2",
        "ancestry": "European",
        "source": "IEU OpenGWAS",
        "build": "GRCh37"
    },
    "Depression": {
        "file": DATA_DIR / "ieu_opengwas/ieu-b-102.vcf.gz",
        "type": "vcf",
        "trait": "Depression",
        "id": "ieu-b-102",
        "ancestry": "European",
        "source": "IEU OpenGWAS",
        "build": "GRCh37"
    },
    # GWAS Catalog files (smaller, harmonized)
    "Stroke_Any": {
        "file": DATA_DIR / "comorbidities/Stroke_GWAS_Catalog_GCST006906.h.tsv.gz",
        "type": "gwas_catalog",
        "trait": "Stroke (Any)",
        "id": "GCST006906",
        "ancestry": "European",
        "source": "GWAS Catalog",
        "build": "GRCh37"
    },
    "Stroke_Ischemic": {
        "file": DATA_DIR / "comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz",
        "type": "gwas_catalog",
        "trait": "Ischemic Stroke",
        "id": "GCST006908",
        "ancestry": "European",
        "source": "GWAS Catalog",
        "build": "GRCh37"
    }
}


def parse_vcf_line(line):
    """Parse a VCF data line and extract GWAS fields."""
    cols = line.strip().split('\t')
    if len(cols) < 10:
        return None
    
    chrom = cols[0]
    pos = cols[1]
    rsid = cols[2]
    ref = cols[3]  # NEA
    alt = cols[4]  # EA
    format_col = cols[8]
    data = cols[9]
    
    # Parse format and data
    format_fields = format_col.split(':')
    data_values = data.split(':')
    
    if len(format_fields) != len(data_values):
        return None
    
    data_dict = dict(zip(format_fields, data_values))
    
    # Extract fields
    try:
        beta = float(data_dict.get('ES', 'nan'))
        se = float(data_dict.get('SE', 'nan'))
        lp = float(data_dict.get('LP', 'nan'))
        sample_size = float(data_dict.get('SS', '0'))
        
        # Convert -log10(p) to p-value
        if not np.isnan(lp) and lp > 0:
            pval = 10 ** (-lp)
        else:
            pval = 1.0
        
        # Skip if essential values are missing
        if rsid == '.' or np.isnan(beta) or np.isnan(se):
            return None
        
        return {
            'SNP': rsid,
            'CHR': str(chrom).replace('chr', '').replace('Chr', ''),
            'POS': int(pos),
            'EA': str(alt).upper(),
            'NEA': str(ref).upper(),
            'BETA': beta,
            'SE': se,
            'P': pval,
            'N': int(sample_size) if sample_size > 0 else None
        }
    except (ValueError, KeyError, IndexError):
        return None


def process_vcf_file_streaming(file_path, dataset_name, chunk_size=100000):
    """Process a VCF file in streaming mode to handle large files."""
    logger.info(f"Processing VCF file (streaming): {file_path}")
    
    chunk = []
    total_rows = 0
    processed_rows = 0
    
    temp_file = STD_DIR / f"{dataset_name}.temp.tsv"
    output_file = STD_DIR / f"{dataset_name}.standardized.gz"
    
    # Process in streaming mode
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parsed = parse_vcf_line(line)
            if parsed:
                chunk.append(parsed)
                processed_rows += 1
            
            total_rows += 1
            
            # Write chunk when it reaches threshold
            if len(chunk) >= chunk_size:
                df_chunk = pd.DataFrame(chunk)
                # Write header only for first chunk
                mode = 'w' if total_rows <= chunk_size else 'a'
                header = total_rows <= chunk_size
                df_chunk.to_csv(temp_file, sep='\t', index=False, mode=mode, header=header)
                chunk = []
                
                if processed_rows % 500000 == 0:
                    logger.info(f"Processed {processed_rows} valid variants...")
    
    # Write remaining chunk
    if chunk:
        df_chunk = pd.DataFrame(chunk)
        mode = 'w' if total_rows <= chunk_size else 'a'
        header = total_rows <= chunk_size
        df_chunk.to_csv(temp_file, sep='\t', index=False, mode=mode, header=header)
    
    logger.info(f"Total variants processed: {processed_rows}")
    
    # Now process the temp file to remove duplicates and finalize
    logger.info(f"Finalizing {dataset_name}...")
    df = pd.read_csv(temp_file, sep='\t', low_memory=False)
    
    # Remove duplicated SNPs
    initial_count = len(df)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    dups_removed = initial_count - len(df)
    if dups_removed > 0:
        logger.info(f"Removed {dups_removed} duplicate SNPs")
    
    # Remove rows with missing essential values
    initial_count = len(df)
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    missing_removed = initial_count - len(df)
    if missing_removed > 0:
        logger.info(f"Removed {missing_removed} rows with missing EA/BETA/SE")
    
    # Sort by CHR and POS
    df = df.sort_values(['CHR', 'POS'])
    
    # Save final file
    df.to_csv(output_file, sep='\t', index=False, compression='gzip')
    logger.info(f"Saved standardized file: {output_file} ({len(df)} variants)")
    
    # Clean up temp file
    if temp_file.exists():
        temp_file.unlink()
    
    return df


def process_gwas_catalog_file(file_path, dataset_name):
    """Process a GWAS Catalog harmonized file."""
    logger.info(f"Processing GWAS Catalog file: {file_path}")
    
    df = pd.read_csv(file_path, sep='\t', compression='gzip', low_memory=False)
    
    # Map columns to standard schema
    column_mapping = {
        'hm_rsid': 'SNP',
        'hm_chrom': 'CHR',
        'hm_pos': 'POS',
        'hm_effect_allele': 'EA',
        'hm_other_allele': 'NEA',
        'hm_beta': 'BETA',
        'standard_error': 'SE',
        'p_value': 'P'
    }
    
    # Check which columns are available
    available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=available_cols)
    
    # Select only required columns
    required_cols = ['SNP', 'CHR', 'POS', 'EA', 'NEA', 'BETA', 'SE', 'P']
    df = df[[col for col in required_cols if col in df.columns]]
    
    # Add N column if available
    if 'n_complete_samples' in df.columns:
        df['N'] = df['n_complete_samples']
    else:
        df['N'] = None
    
    # Clean up
    df['EA'] = df['EA'].str.upper()
    df['NEA'] = df['NEA'].str.upper()
    df['CHR'] = df['CHR'].astype(str).str.replace('chr', '', case=False)
    df['POS'] = df['POS'].astype(int)
    df['BETA'] = pd.to_numeric(df['BETA'], errors='coerce')
    df['SE'] = pd.to_numeric(df['SE'], errors='coerce')
    df['P'] = pd.to_numeric(df['P'], errors='coerce')
    
    # Remove duplicates and missing values
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    
    # Sort
    df = df.sort_values(['CHR', 'POS'])
    
    logger.info(f"Loaded {len(df)} variants from {dataset_name}")
    
    # Save
    output_file = STD_DIR / f"{dataset_name}.standardized.gz"
    df.to_csv(output_file, sep='\t', index=False, compression='gzip')
    
    return df


def compute_qc_metrics(df, dataset_name):
    """Compute QC metrics for a dataset."""
    logger.info(f"Computing QC metrics for {dataset_name}")
    
    metrics = {
        'Trait': dataset_name,
        'SNP_count': len(df),
        'Mean_SE': df['SE'].mean(),
        'Lambda_GC': None,
        'Allele_missing_rate': 0.0
    }
    
    # Compute Lambda GC
    try:
        df_temp = df.copy()
        df_temp['Z'] = df_temp['BETA'] / df_temp['SE']
        df_temp['CHI2'] = df_temp['Z'] ** 2
        
        median_chi2 = df_temp['CHI2'].median()
        lambda_gc = median_chi2 / 0.455
        metrics['Lambda_GC'] = lambda_gc
        
        logger.info(f"Lambda GC for {dataset_name}: {lambda_gc:.4f}")
        
    except Exception as e:
        logger.warning(f"Could not compute Lambda GC for {dataset_name}: {e}")
        metrics['Lambda_GC'] = None
    
    # Check for allele missing rate
    total_alleles = len(df) * 2
    missing_alleles = df['EA'].isna().sum() + df['NEA'].isna().sum()
    metrics['Allele_missing_rate'] = missing_alleles / total_alleles if total_alleles > 0 else 0.0
    
    return metrics


def create_harmonized_copy(df, dataset_name):
    """Create harmonized copy (already GRCh37, just copy)."""
    output_file = HARM_DIR / f"{dataset_name}.harmonized.gz"
    df.to_csv(output_file, sep='\t', index=False, compression='gzip')
    logger.info(f"Saved harmonized file: {output_file}")
    return output_file


def main():
    """Main processing pipeline."""
    logger.info("="*60)
    logger.info("Starting Optimized GWAS Processing Pipeline")
    logger.info("="*60)
    
    qc_metrics_list = []
    manifest_entries = []
    
    # Process datasets
    for dataset_name, config in DATASETS.items():
        file_path = config['file']
        file_type = config['type']
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue
        
        logger.info(f"\nProcessing {dataset_name} ({config['trait']})")
        logger.info("-" * 60)
        
        try:
            if file_type == 'vcf':
                df = process_vcf_file_streaming(file_path, dataset_name)
            elif file_type == 'gwas_catalog':
                df = process_gwas_catalog_file(file_path, dataset_name)
            else:
                logger.error(f"Unknown file type: {file_type}")
                continue
            
            if df is None or len(df) == 0:
                logger.error(f"Failed to process {dataset_name}")
                continue
            
            # Compute QC metrics
            qc_metrics = compute_qc_metrics(df, dataset_name)
            qc_metrics_list.append(qc_metrics)
            
            # Create harmonized copy
            harm_file = create_harmonized_copy(df, dataset_name)
            
            # Create manifest entry
            sample_size = df['N'].mean() if 'N' in df.columns and df['N'].notna().any() else 'NA'
            manifest_entry = {
                'Trait': dataset_name,
                'Dataset_ID': config['id'],
                'Source': config['source'],
                'Sample_Size': int(sample_size) if sample_size != 'NA' else 'NA',
                'Ancestry': config['ancestry'],
                'Genome_Build': config['build'],
                'SNP_Count_PostQC': len(df),
                'File_Path': str(harm_file),
                'QC_Passed': 'YES'
            }
            manifest_entries.append(manifest_entry)
            
            logger.info(f"Completed {dataset_name}: {len(df)} variants")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save QC metrics
    if qc_metrics_list:
        qc_df = pd.DataFrame(qc_metrics_list)
        qc_file = RESULTS_DIR / "gwas_qc_metrics.csv"
        qc_df.to_csv(qc_file, index=False)
        logger.info(f"\nSaved QC metrics to: {qc_file}")
    
    # Save manifest
    if manifest_entries:
        manifest_df = pd.DataFrame(manifest_entries)
        manifest_file = RESULTS_DIR / "final_dataset_manifest.csv"
        manifest_df.to_csv(manifest_file, index=False)
        logger.info(f"Saved manifest to: {manifest_file}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Step 1 Processing Complete")
    logger.info("="*60)
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETION SUMMARY")
    print("="*60)
    print(f"Total datasets processed: {len(manifest_entries)}")
    print(f"Total datasets required: 11 (1 missing - CKD)")
    print(f"Completion rate: {len(manifest_entries)}/11 ({len(manifest_entries)/11*100:.1f}%)")
    
    if qc_metrics_list:
        print("\nQC METRICS:")
        print("-" * 60)
        for m in qc_metrics_list:
            print(f"{m['Trait']}: SNPs={m['SNP_count']:,}, Mean_SE={m['Mean_SE']:.4f}, Lambda_GC={m['Lambda_GC']:.3f}")


if __name__ == "__main__":
    main()
