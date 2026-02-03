#!/usr/bin/env python3
"""
Post-process T2D and process remaining datasets:
- Sort and deduplicate T2D
- Process AD (ieu-b-2.vcf.gz)
- Process Stroke_Any and Stroke_Ischemic (TSV files)
"""

import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys

DATA_DIR = Path("data")
STD_DIR = DATA_DIR / "gwas_standardized"
HARM_DIR = DATA_DIR / "harmonized"

def post_process_t2d():
    """Sort, deduplicate, and create harmonized version of T2D."""
    print("\n" + "="*60)
    print("Post-processing T2D: Sorting and deduplicating...")
    print("="*60)
    
    # Read T2D
    t2d_file = STD_DIR / "T2D.standardized.gz"
    print(f"Reading {t2d_file}...")
    df = pd.read_csv(t2d_file, sep='\t', compression='gzip')
    
    print(f"  Initial rows: {len(df):,}")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    print(f"  Duplicates removed: {before - len(df):,}")
    
    # Remove missing
    before = len(df)
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    print(f"  Missing values removed: {before - len(df):,}")
    
    # Sort
    df = df.sort_values(['CHR', 'POS'])
    
    print(f"  Final variants: {len(df):,}")
    
    # Save
    df.to_csv(STD_DIR / "T2D.standardized.gz", sep='\t', index=False, compression='gzip')
    df.to_csv(HARM_DIR / "T2D.harmonized.gz", sep='\t', index=False, compression='gzip')
    
    print("  ✓ T2D standardized and harmonized files saved")
    return len(df)

def process_ad():
    """Process AD VCF file."""
    print("\n" + "="*60)
    print("Processing AD (ieu-b-2.vcf.gz)...")
    print("="*60)
    
    vcf_file = DATA_DIR / "ieu_opengwas/ieu-b-2.vcf.gz"
    rows = []
    
    with gzip.open(vcf_file, 'rt') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            
            cols = line.strip().split('\t')
            if len(cols) < 10:
                continue
            
            try:
                chrom, pos, rsid, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
                format_col, data = cols[8], cols[9]
                
                format_fields = format_col.split(':')
                data_values = data.split(':')
                
                if len(format_fields) != len(data_values):
                    continue
                
                data_dict = dict(zip(format_fields, data_values))
                
                beta = float(data_dict.get('ES', 'nan'))
                se = float(data_dict.get('SE', 'nan'))
                lp = float(data_dict.get('LP', 'nan'))
                sample_size = float(data_dict.get('SS', '0'))
                
                if np.isnan(beta) or np.isnan(se) or rsid == '.':
                    continue
                
                pval = 10 ** (-lp) if not np.isnan(lp) and lp > 0 else 1.0
                
                rows.append({
                    'SNP': rsid,
                    'CHR': str(chrom).replace('chr', ''),
                    'POS': int(pos),
                    'EA': str(alt).upper(),
                    'NEA': str(ref).upper(),
                    'BETA': beta,
                    'SE': se,
                    'P': pval,
                    'N': int(sample_size) if sample_size > 0 else None
                })
                
            except:
                continue
            
            if i % 500000 == 0 and i > 0:
                print(f"  ... processed {i:,} lines, {len(rows):,} variants")
    
    print(f"  Total variants: {len(rows):,}")
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    df = df.sort_values(['CHR', 'POS'])
    
    print(f"  Final variants after cleaning: {len(df):,}")
    
    df.to_csv(STD_DIR / "AD.standardized.gz", sep='\t', index=False, compression='gzip')
    df.to_csv(HARM_DIR / "AD.harmonized.gz", sep='\t', index=False, compression='gzip')
    
    print("  ✓ AD standardized and harmonized files saved")
    return len(df)

def process_stroke_any():
    """Process Stroke_Any TSV file."""
    print("\n" + "="*60)
    print("Processing Stroke_Any (GCST006906.h.tsv.gz)...")
    print("="*60)
    
    tsv_file = DATA_DIR / "comorbidities/Stroke_GWAS_Catalog_GCST006906.h.tsv.gz"
    rows = []
    
    with gzip.open(tsv_file, 'rt', errors='replace') as f:
        header = f.readline().strip().split('\t')
        
        for i, line in enumerate(f):
            try:
                cols = line.strip().split('\t')
                if len(cols) < 10:
                    continue
                
                row = dict(zip(header, cols))
                
                rsid = row.get('hm_rsid', '')
                if not rsid or rsid == 'NA':
                    continue
                
                beta = row.get('hm_beta', '')
                se = row.get('standard_error', '')
                pval = row.get('p_value', '')
                
                if beta in ['', 'NA'] or se in ['', 'NA']:
                    continue
                
                rows.append({
                    'SNP': rsid,
                    'CHR': str(row.get('hm_chrom', '')).replace('chr', ''),
                    'POS': int(row.get('hm_pos', 0)),
                    'EA': str(row.get('hm_effect_allele', '')).upper(),
                    'NEA': str(row.get('hm_other_allele', '')).upper(),
                    'BETA': float(beta),
                    'SE': float(se),
                    'P': float(pval) if pval not in ['', 'NA'] else 1.0,
                    'N': None
                })
                
            except (ValueError, IndexError):
                continue
            
            if i % 500000 == 0 and i > 0:
                print(f"  ... processed {i:,} lines, {len(rows):,} variants")
            
            if i > 8000000:
                print(f"  Warning: Reached safety limit")
                break
    
    print(f"  Total variants: {len(rows):,}")
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    df = df.sort_values(['CHR', 'POS'])
    
    print(f"  Final variants after cleaning: {len(df):,}")
    
    df.to_csv(STD_DIR / "Stroke_Any.standardized.gz", sep='\t', index=False, compression='gzip')
    df.to_csv(HARM_DIR / "Stroke_Any.harmonized.gz", sep='\t', index=False, compression='gzip')
    
    print("  ✓ Stroke_Any standardized and harmonized files saved")
    return len(df)

def process_stroke_ischemic():
    """Process Stroke_Ischemic TSV file."""
    print("\n" + "="*60)
    print("Processing Stroke_Ischemic (GCST006908.h.tsv.gz)...")
    print("="*60)
    
    tsv_file = DATA_DIR / "comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz"
    rows = []
    
    with gzip.open(tsv_file, 'rt', errors='replace') as f:
        header = f.readline().strip().split('\t')
        
        for i, line in enumerate(f):
            try:
                cols = line.strip().split('\t')
                if len(cols) < 10:
                    continue
                
                row = dict(zip(header, cols))
                
                rsid = row.get('hm_rsid', '')
                if not rsid or rsid == 'NA':
                    continue
                
                beta = row.get('hm_beta', '')
                se = row.get('standard_error', '')
                pval = row.get('p_value', '')
                
                if beta in ['', 'NA'] or se in ['', 'NA']:
                    continue
                
                rows.append({
                    'SNP': rsid,
                    'CHR': str(row.get('hm_chrom', '')).replace('chr', ''),
                    'POS': int(row.get('hm_pos', 0)),
                    'EA': str(row.get('hm_effect_allele', '')).upper(),
                    'NEA': str(row.get('hm_other_allele', '')).upper(),
                    'BETA': float(beta),
                    'SE': float(se),
                    'P': float(pval) if pval not in ['', 'NA'] else 1.0,
                    'N': None
                })
                
            except (ValueError, IndexError):
                continue
            
            if i % 500000 == 0 and i > 0:
                print(f"  ... processed {i:,} lines, {len(rows):,} variants")
            
            if i > 8000000:
                print(f"  Warning: Reached safety limit")
                break
    
    print(f"  Total variants: {len(rows):,}")
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    df = df.sort_values(['CHR', 'POS'])
    
    print(f"  Final variants after cleaning: {len(df):,}")
    
    df.to_csv(STD_DIR / "Stroke_Ischemic.standardized.gz", sep='\t', index=False, compression='gzip')
    df.to_csv(HARM_DIR / "Stroke_Ischemic.harmonized.gz", sep='\t', index=False, compression='gzip')
    
    print("  ✓ Stroke_Ischemic standardized and harmonized files saved")
    return len(df)

def main():
    print("="*60)
    print("GWAS DATA PROCESSING - Remaining 3 Datasets + T2D Post-process")
    print("="*60)
    
    results = {}
    
    # 1. Post-process T2D (sort and deduplicate)
    results['T2D'] = post_process_t2d()
    
    # 2. Process AD
    results['AD'] = process_ad()
    
    # 3. Process Stroke_Any
    results['Stroke_Any'] = process_stroke_any()
    
    # 4. Process Stroke_Ischemic
    results['Stroke_Ischemic'] = process_stroke_ischemic()
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*60)
    print("\nFinal variant counts:")
    for name, count in results.items():
        print(f"  {name:20s}: {count:>10,} variants")
    
    print("\nGenerated files:")
    for f in sorted(STD_DIR.glob("*.standardized.gz")):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name:40s} ({size_mb:>6.1f} MB)")
    
    print("\nHarmonized files:")
    for f in sorted(HARM_DIR.glob("*.harmonized.gz")):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name:40s} ({size_mb:>6.1f} MB)")
    
    print("\n" + "="*60)
    print("All 4 datasets processed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
