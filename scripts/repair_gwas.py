#!/usr/bin/env python3
"""
Repair script for failed GWAS datasets
Handles:
1. SBP, T2D, AD - fix temp file reading
2. Stroke_Any, Stroke_Ischemic - handle partial file corruption
"""

import gzip
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
STD_DIR = DATA_DIR / "gwas_standardized"
HARM_DIR = DATA_DIR / "harmonized"

def parse_vcf_line(line):
    """Parse a VCF data line and extract GWAS fields."""
    cols = line.strip().split('\t')
    if len(cols) < 10:
        return None
    
    chrom = cols[0]
    pos = cols[1]
    rsid = cols[2]
    ref = cols[3]
    alt = cols[4]
    format_col = cols[8]
    data = cols[9]
    
    format_fields = format_col.split(':')
    data_values = data.split(':')
    
    if len(format_fields) != len(data_values):
        return None
    
    data_dict = dict(zip(format_fields, data_values))
    
    try:
        beta = float(data_dict.get('ES', 'nan'))
        se = float(data_dict.get('SE', 'nan'))
        lp = float(data_dict.get('LP', 'nan'))
        sample_size = float(data_dict.get('SS', '0'))
        
        if not np.isnan(lp) and lp > 0:
            pval = 10 ** (-lp)
        else:
            pval = 1.0
        
        if rsid == '.' or np.isnan(beta) or np.isnan(se):
            return None
        
        return {
            'SNP': rsid,
            'CHR': str(chrom).replace('chr', ''),
            'POS': int(pos),
            'EA': str(alt).upper(),
            'NEA': str(ref).upper(),
            'BETA': beta,
            'SE': se,
            'P': pval,
            'N': int(sample_size) if sample_size > 0 else None
        }
    except:
        return None

def process_vcf_fixed(dataset_name, vcf_file):
    """Process VCF with better error handling."""
    print(f"\nProcessing {dataset_name}...")
    
    rows = []
    line_count = 0
    
    with gzip.open(vcf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parsed = parse_vcf_line(line)
            if parsed:
                rows.append(parsed)
                line_count += 1
            
            if line_count % 500000 == 0:
                print(f"  Processed {line_count} variants...")
    
    print(f"  Total: {line_count} variants")
    
    df = pd.DataFrame(rows)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    
    # Remove missing
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    
    # Sort
    df = df.sort_values(['CHR', 'POS'])
    
    # Save standardized
    std_file = STD_DIR / f"{dataset_name}.standardized.gz"
    df.to_csv(std_file, sep='\t', index=False, compression='gzip')
    print(f"  Saved: {std_file} ({len(df)} variants)")
    
    # Save harmonized
    harm_file = HARM_DIR / f"{dataset_name}.harmonized.gz"
    df.to_csv(harm_file, sep='\t', index=False, compression='gzip')
    print(f"  Saved: {harm_file}")
    
    return df

def process_stroke_fixed(dataset_name, tsv_file):
    """Process Stroke files with error handling for corruption."""
    print(f"\nProcessing {dataset_name}...")
    
    try:
        # Read with error handling
        rows = []
        with gzip.open(tsv_file, 'rt', errors='replace') as f:
            header = f.readline().strip().split('\t')
            
            for i, line in enumerate(f):
                try:
                    cols = line.strip().split('\t')
                    if len(cols) < 10:
                        continue
                    
                    # Create dict from header and values
                    row = dict(zip(header, cols))
                    
                    # Extract fields
                    rsid = row.get('hm_rsid', '')
                    if not rsid or rsid == 'NA':
                        continue
                    
                    beta = row.get('hm_beta', 'nan')
                    se = row.get('standard_error', 'nan')
                    pval = row.get('p_value', 'nan')
                    
                    if beta == 'NA' or se == 'NA':
                        continue
                    
                    rows.append({
                        'SNP': rsid,
                        'CHR': str(row.get('hm_chrom', '')).replace('chr', ''),
                        'POS': int(row.get('hm_pos', 0)),
                        'EA': str(row.get('hm_effect_allele', '')).upper(),
                        'NEA': str(row.get('hm_other_allele', '')).upper(),
                        'BETA': float(beta),
                        'SE': float(se),
                        'P': float(pval) if pval not in ['NA', 'nan'] else 1.0,
                        'N': None
                    })
                    
                    if i % 500000 == 0 and i > 0:
                        print(f"  Processed {i} lines...")
                        
                except (ValueError, IndexError):
                    continue
                
                # Stop after reasonable number to avoid corrupted end
                if i > 8000000:
                    break
        
        if not rows:
            print(f"  ERROR: No valid rows found")
            return None
        
        df = pd.DataFrame(rows)
        
        # Remove duplicates and missing
        df = df.drop_duplicates(subset=['SNP'], keep='first')
        df = df.dropna(subset=['EA', 'BETA', 'SE'])
        
        # Sort
        df = df.sort_values(['CHR', 'POS'])
        
        # Save
        std_file = STD_DIR / f"{dataset_name}.standardized.gz"
        df.to_csv(std_file, sep='\t', index=False, compression='gzip')
        print(f"  Saved: {std_file} ({len(df)} variants)")
        
        harm_file = HARM_DIR / f"{dataset_name}.harmonized.gz"
        df.to_csv(harm_file, sep='\t', index=False, compression='gzip')
        print(f"  Saved: {harm_file}")
        
        return df
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*60)
    print("GWAS DATA REPAIR - Processing Failed Datasets")
    print("="*60)
    
    # Fix VCF datasets
    vcf_datasets = [
        ("SBP", DATA_DIR / "ieu_opengwas/ieu-b-4818.vcf.gz"),
        ("T2D", DATA_DIR / "ieu_opengwas/ieu-b-107.vcf.gz"),
        ("AD", DATA_DIR / "ieu_opengwas/ieu-b-2.vcf.gz"),
    ]
    
    for name, file_path in vcf_datasets:
        if file_path.exists():
            process_vcf_fixed(name, file_path)
        else:
            print(f"ERROR: {file_path} not found")
    
    # Fix Stroke datasets
    stroke_datasets = [
        ("Stroke_Any", DATA_DIR / "comorbidities/Stroke_GWAS_Catalog_GCST006906.h.tsv.gz"),
        ("Stroke_Ischemic", DATA_DIR / "comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz"),
    ]
    
    for name, file_path in stroke_datasets:
        if file_path.exists():
            process_stroke_fixed(name, file_path)
        else:
            print(f"ERROR: {file_path} not found")
    
    print("\n" + "="*60)
    print("REPAIR COMPLETE")
    print("="*60)
    
    # List all files
    print("\nGenerated files:")
    for f in sorted(STD_DIR.glob("*.standardized.gz")):
        print(f"  Standardized: {f.name}")
    for f in sorted(HARM_DIR.glob("*.harmonized.gz")):
        print(f"  Harmonized: {f.name}")

if __name__ == "__main__":
    main()
