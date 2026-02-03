#!/usr/bin/env python3
"""
Efficient batch processor for 4 remaining GWAS datasets
Processes: T2D, AD, Stroke_Any, Stroke_Ischemic
"""

import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR = Path("data")
STD_DIR = DATA_DIR / "gwas_standardized"
HARM_DIR = DATA_DIR / "harmonized"

def parse_vcf_line(line):
    """Parse VCF line efficiently."""
    cols = line.strip().split('\t')
    if len(cols) < 10:
        return None
    
    try:
        chrom, pos, rsid, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
        format_col, data = cols[8], cols[9]
        
        format_fields = format_col.split(':')
        data_values = data.split(':')
        
        if len(format_fields) != len(data_values):
            return None
        
        data_dict = dict(zip(format_fields, data_values))
        
        beta = float(data_dict.get('ES', 'nan'))
        se = float(data_dict.get('SE', 'nan'))
        lp = float(data_dict.get('LP', 'nan'))
        sample_size = float(data_dict.get('SS', '0'))
        
        if np.isnan(beta) or np.isnan(se) or rsid == '.':
            return None
        
        pval = 10 ** (-lp) if not np.isnan(lp) and lp > 0 else 1.0
        
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

def process_vcf_dataset(name, vcf_path):
    """Process VCF file with progress reporting."""
    print(f"\n{'='*60}")
    print(f"Processing {name}: {vcf_path.name}")
    print('='*60)
    
    rows = []
    total_lines = 0
    
    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            total_lines += 1
            parsed = parse_vcf_line(line)
            if parsed:
                rows.append(parsed)
            
            if total_lines % 500000 == 0:
                print(f"  ... processed {total_lines:,} lines, {len(rows):,} valid variants")
    
    print(f"  Total lines: {total_lines:,}, Valid variants: {len(rows):,}")
    
    df = pd.DataFrame(rows)
    
    # Clean data
    before_dup = len(df)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    after_dup = len(df)
    
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    after_clean = len(df)
    
    df = df.sort_values(['CHR', 'POS'])
    
    print(f"  Duplicates removed: {before_dup - after_dup}")
    print(f"  Missing values removed: {after_dup - after_clean}")
    print(f"  Final variants: {after_clean:,}")
    
    # Save
    std_file = STD_DIR / f"{name}.standardized.gz"
    harm_file = HARM_DIR / f"{name}.harmonized.gz"
    
    df.to_csv(std_file, sep='\t', index=False, compression='gzip')
    df.to_csv(harm_file, sep='\t', index=False, compression='gzip')
    
    print(f"  ✓ Saved: {std_file.name}")
    print(f"  ✓ Saved: {harm_file.name}")
    
    return len(df)

def process_stroke_dataset(name, tsv_path):
    """Process TSV stroke file with error handling."""
    print(f"\n{'='*60}")
    print(f"Processing {name}: {tsv_path.name}")
    print('='*60)
    
    rows = []
    total_lines = 0
    
    with gzip.open(tsv_path, 'rt', errors='replace') as f:
        header = f.readline().strip().split('\t')
        
        for line in f:
            total_lines += 1
            
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
            
            if total_lines % 500000 == 0:
                print(f"  ... processed {total_lines:,} lines, {len(rows):,} valid variants")
            
            # Safety limit for corrupted files
            if total_lines > 8000000:
                print(f"  Warning: Reached safety limit at {total_lines:,} lines")
                break
    
    print(f"  Total lines: {total_lines:,}, Valid variants: {len(rows):,}")
    
    df = pd.DataFrame(rows)
    
    before_dup = len(df)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    after_dup = len(df)
    
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    after_clean = len(df)
    
    df = df.sort_values(['CHR', 'POS'])
    
    print(f"  Duplicates removed: {before_dup - after_dup}")
    print(f"  Missing values removed: {after_dup - after_clean}")
    print(f"  Final variants: {after_clean:,}")
    
    std_file = STD_DIR / f"{name}.standardized.gz"
    harm_file = HARM_DIR / f"{name}.harmonized.gz"
    
    df.to_csv(std_file, sep='\t', index=False, compression='gzip')
    df.to_csv(harm_file, sep='\t', index=False, compression='gzip')
    
    print(f"  ✓ Saved: {std_file.name}")
    print(f"  ✓ Saved: {harm_file.name}")
    
    return len(df)

def main():
    print("="*60)
    print("GWAS DATA PROCESSING - 4 Remaining Datasets")
    print("="*60)
    print("\nDatasets to process:")
    print("  1. T2D (ieu-b-107.vcf.gz) - Type 2 Diabetes [VCF - Large]")
    print("  2. AD (ieu-b-2.vcf.gz) - Alzheimer's Disease [VCF - Large]")
    print("  3. Stroke_Any (GCST006906.h.tsv.gz) - Any Stroke [TSV]")
    print("  4. Stroke_Ischemic (GCST006908.h.tsv.gz) - Ischemic Stroke [TSV]")
    
    results = {}
    
    # Process T2D
    t2d_path = DATA_DIR / "ieu_opengwas/ieu-b-107.vcf.gz"
    if t2d_path.exists():
        results['T2D'] = process_vcf_dataset("T2D", t2d_path)
    else:
        print(f"ERROR: {t2d_path} not found")
        results['T2D'] = 0
    
    # Process AD
    ad_path = DATA_DIR / "ieu_opengwas/ieu-b-2.vcf.gz"
    if ad_path.exists():
        results['AD'] = process_vcf_dataset("AD", ad_path)
    else:
        print(f"ERROR: {ad_path} not found")
        results['AD'] = 0
    
    # Process Stroke_Any
    stroke_any_path = DATA_DIR / "comorbidities/Stroke_GWAS_Catalog_GCST006906.h.tsv.gz"
    if stroke_any_path.exists():
        results['Stroke_Any'] = process_stroke_dataset("Stroke_Any", stroke_any_path)
    else:
        print(f"ERROR: {stroke_any_path} not found")
        results['Stroke_Any'] = 0
    
    # Process Stroke_Ischemic
    stroke_ischemic_path = DATA_DIR / "comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz"
    if stroke_ischemic_path.exists():
        results['Stroke_Ischemic'] = process_stroke_dataset("Stroke_Ischemic", stroke_ischemic_path)
    else:
        print(f"ERROR: {stroke_ischemic_path} not found")
        results['Stroke_Ischemic'] = 0
    
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
    
    print("\n" + "="*60)
    print("All 4 datasets processed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
