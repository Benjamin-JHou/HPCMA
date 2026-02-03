#!/usr/bin/env python3
"""
Quick process for Stroke datasets - simplified version
"""

import gzip
import pandas as pd
from pathlib import Path
import sys

DATA_DIR = Path("data")
STD_DIR = DATA_DIR / "gwas_standardized"
HARM_DIR = DATA_DIR / "harmonized"

def process_stroke_quick(name, tsv_path, out_prefix):
    """Quick process with minimal overhead and EOF handling."""
    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print('='*60)
    
    rows = []
    count = 0
    
    try:
        with gzip.open(tsv_path, 'rt', errors='replace') as f:
            header = f.readline().strip().split('\t')
            
            for line in f:
                count += 1
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
                        'P': float(row.get('p_value', 1)) if row.get('p_value') not in ['', 'NA'] else 1.0,
                        'N': None
                    })
                    
                except:
                    continue
                
                if count % 1000000 == 0:
                    print(f"  {count:,} lines, {len(rows):,} variants")
    except EOFError:
        print(f"  Note: File ended unexpectedly at line {count:,}")
    
    print(f"  Total: {count:,} lines, {len(rows):,} variants")
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    df = df.sort_values(['CHR', 'POS'])
    
    print(f"  Final: {len(df):,} variants")
    
    # Save both
    df.to_csv(STD_DIR / f"{out_prefix}.standardized.gz", sep='\t', index=False, compression='gzip')
    df.to_csv(HARM_DIR / f"{out_prefix}.harmonized.gz", sep='\t', index=False, compression='gzip')
    
    print(f"  ✓ Saved both files")
    return len(df)

print("Processing Stroke datasets...")

# Process both
results = {}

results['Stroke_Any'] = process_stroke_quick(
    "Stroke_Any",
    DATA_DIR / "comorbidities/Stroke_GWAS_Catalog_GCST006906.h.tsv.gz",
    "Stroke_Any"
)

results['Stroke_Ischemic'] = process_stroke_quick(
    "Stroke_Ischemic",
    DATA_DIR / "comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz",
    "Stroke_Ischemic"
)

print("\n" + "="*60)
print("Complete!")
for name, count in results.items():
    print(f"  {name}: {count:,} variants")
