#!/usr/bin/env python3
"""
Process Stroke_Ischemic with inline execution
"""

import gzip
import pandas as pd
from pathlib import Path
import sys

DATA_DIR = Path("data")
STD_DIR = DATA_DIR / "gwas_standardized"
HARM_DIR = DATA_DIR / "harmonized"

print("="*60, file=sys.stderr)
print("Processing Stroke_Ischemic", file=sys.stderr)
print("="*60, file=sys.stderr)

rows = []
count = 0

try:
    with gzip.open(DATA_DIR / "comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz", 'rt', errors='replace') as f:
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
                print(f"  {count:,} lines, {len(rows):,} variants", file=sys.stderr)
                
except EOFError:
    print(f"  Note: File ended unexpectedly at line {count:,}", file=sys.stderr)

print(f"  Total: {count:,} lines, {len(rows):,} variants", file=sys.stderr)

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=['SNP'], keep='first')
df = df.dropna(subset=['EA', 'BETA', 'SE'])
df = df.sort_values(['CHR', 'POS'])

print(f"  Final: {len(df):,} variants", file=sys.stderr)

# Save both
df.to_csv(STD_DIR / "Stroke_Ischemic.standardized.gz", sep='\t', index=False, compression='gzip')
df.to_csv(HARM_DIR / "Stroke_Ischemic.harmonized.gz", sep='\t', index=False, compression='gzip')

print(f"  ✓ Saved both files", file=sys.stderr)
print("="*60, file=sys.stderr)
print(f"Stroke_Ischemic: {len(df):,} variants", file=sys.stderr)
print("="*60, file=sys.stderr)

# Also print to stdout for verification
print(f"Stroke_Ischemic: {len(df):,} variants")
