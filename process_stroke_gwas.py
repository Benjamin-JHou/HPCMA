#!/usr/bin/env python3
"""Process Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz - Quick version"""
import gzip
import pandas as pd
import numpy as np
from statistics import NormalDist

def process_file():
    input_file = 'data/comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz'
    out_std = 'data/gwas_standardized/Stroke_Ischemic_GCST006908.txt.gz'
    out_harm = 'data/harmonized/Stroke_Ischemic_GCST006908.txt.gz'
    
    print(f"Reading {input_file}...")
    
    # Read with error handling
    lines = []
    try:
        with gzip.open(input_file, 'rt', encoding='utf-8', errors='replace') as f:
            header = f.readline().strip().split('\t')
            print(f"Columns: {len(header)}")
            
            # Get column indices
            idx = {c: header.index(c) for c in header}
            
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    print(f"  Line {i:,}...", end='\r')
                try:
                    if line.strip():
                        lines.append(line.strip().split('\t'))
                except:
                    break
    except Exception as e:
        print(f"\nEOF error after {len(lines):,} lines: {e}")
    
    print(f"\nTotal lines read: {len(lines):,}")
    
    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame(lines, columns=header)
    
    # Standardize
    print("Standardizing...")
    std = pd.DataFrame({
        'SNP': df['hm_rsid'],
        'CHR': pd.to_numeric(df['hm_chrom'], errors='coerce'),
        'POS': pd.to_numeric(df['hm_pos'], errors='coerce'),
        'EA': df['hm_effect_allele'],
        'NEA': df['hm_other_allele'],
        'BETA': pd.to_numeric(df['hm_beta'], errors='coerce'),
        'SE': pd.to_numeric(df['standard_error'], errors='coerce'),
        'P': pd.to_numeric(df['p_value'], errors='coerce'),
        'EAF': pd.to_numeric(df['hm_effect_allele_frequency'], errors='coerce')
    })
    
    # Clean
    std = std.dropna(subset=['SNP', 'CHR', 'POS', 'BETA', 'SE', 'P'])
    print(f"Clean rows: {len(std):,}")
    
    # Save standardized
    print(f"Saving to {out_std}...")
    std.to_csv(out_std, sep='\t', index=False, compression='gzip')
    
    # Save harmonized
    print(f"Saving to {out_harm}...")
    harm = std.copy()
    harm['N'] = np.nan
    harm.to_csv(out_harm, sep='\t', index=False, compression='gzip')
    
    # QC metrics
    print("\n" + "="*50)
    print("QC METRICS")
    print("="*50)
    print(f"SNP count: {len(std):,}")
    print(f"Mean SE: {std['SE'].mean():.6f}")
    
    # Lambda GC
    p = std['P'][(std['P'] > 0) & (std['P'] < 1)]
    p_samp = np.random.choice(p, min(1000000, len(p)), replace=False) if len(p) > 1000000 else p
    z = np.array([NormalDist().inv_cdf(1 - x/2) for x in p_samp])
    chi2 = z ** 2
    print(f"Lambda GC: {np.median(chi2)/0.455:.4f}")
    
    # Missing rate
    miss = (std['EA'].isna().sum() + std['NEA'].isna().sum()) / (len(std) * 2)
    print(f"Allele missing rate: {miss:.6f}")
    print(f"EAF available: {std['EAF'].notna().sum():,} ({std['EAF'].notna().sum()/len(std)*100:.2f}%)")
    print("="*50)
    print("Done!")

if __name__ == '__main__':
    process_file()
