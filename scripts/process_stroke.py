#!/usr/bin/env python3
"""
Process Stroke datasets with corruption handling
"""

import gzip
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
STD_DIR = DATA_DIR / "gwas_standardized"
HARM_DIR = DATA_DIR / "harmonized"

def process_stroke_file(name, tsv_path, std_file, harm_file):
    """Process stroke TSV with error handling."""
    print(f"\n{'='*60}")
    print(f"Processing {name}: {tsv_path.name}")
    print(f"  Input: {tsv_path.stat().st_size / (1024**2):.1f} MB")
    print('='*60)
    
    rows = []
    line_count = 0
    error_count = 0
    
    try:
        with gzip.open(tsv_path, 'rt', errors='replace') as f:
            header = f.readline().strip().split('\t')
            
            for line in f:
                line_count += 1
                
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
                    
                except (ValueError, IndexError):
                    continue
                
                if line_count % 500000 == 0:
                    print(f"  ... {line_count:,} lines, {len(rows):,} variants")
                
                if line_count > 8000000:
                    break
                    
    except EOFError as e:
        print(f"  Note: File ended unexpectedly at line {line_count:,}")
        print(f"  Processing {len(rows):,} collected variants...")
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"  Total: {line_count:,} lines, {len(rows):,} valid variants")
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    df = df.sort_values(['CHR', 'POS'])
    
    print(f"  After cleaning: {len(df):,} variants")
    
    df.to_csv(std_file, sep='\t', index=False, compression='gzip')
    df.to_csv(harm_file, sep='\t', index=False, compression='gzip')
    
    print(f"  ✓ Saved: {std_file.name} ({std_file.stat().st_size / (1024**2):.1f} MB)")
    print(f"  ✓ Saved: {harm_file.name} ({harm_file.stat().st_size / (1024**2):.1f} MB)")
    
    return len(df)

print("="*60)
print("Processing Stroke Datasets (with corruption handling)")
print("="*60)

# Process Stroke_Any
results = {}
results['Stroke_Any'] = process_stroke_file(
    "Stroke_Any",
    DATA_DIR / "comorbidities/Stroke_GWAS_Catalog_GCST006906.h.tsv.gz",
    STD_DIR / "Stroke_Any.standardized.gz",
    HARM_DIR / "Stroke_Any.harmonized.gz"
)

# Process Stroke_Ischemic
results['Stroke_Ischemic'] = process_stroke_file(
    "Stroke_Ischemic",
    DATA_DIR / "comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz",
    STD_DIR / "Stroke_Ischemic.standardized.gz",
    HARM_DIR / "Stroke_Ischemic.harmonized.gz"
)

print("\n" + "="*60)
print("Stroke Processing Complete")
print("="*60)
for name, count in results.items():
    print(f"  {name}: {count:,} variants")
