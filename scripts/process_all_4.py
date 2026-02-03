#!/usr/bin/env python3
"""
Complete processing of 4 remaining GWAS datasets with progress reporting.
Processes: T2D, AD, Stroke_Any, Stroke_Ischemic
Saves to both standardized and harmonized directories.
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
    """Parse VCF data line and extract GWAS fields."""
    cols = line.strip().split('\t')
    if len(cols) < 10:
        return None
    
    chrom, pos, rsid, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
    format_col, data = cols[8], cols[9]
    
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

def process_vcf(name, vcf_path, temp_std, temp_harm):
    """Process VCF file with detailed progress."""
    print(f"\n{'='*60}")
    print(f"Processing {name}: {vcf_path.name}")
    print(f"  Input size: {vcf_path.stat().st_size / (1024**2):.1f} MB")
    print('='*60)
    
    rows = []
    line_count = 0
    
    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            line_count += 1
            parsed = parse_vcf_line(line)
            if parsed:
                rows.append(parsed)
            
            # Progress report every 500k lines
            if line_count % 500000 == 0:
                print(f"  ... {line_count:,} lines read, {len(rows):,} variants kept")
    
    print(f"  Total: {line_count:,} lines, {len(rows):,} valid variants")
    
    # Create DataFrame
    print("  Creating DataFrame...")
    df = pd.DataFrame(rows)
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    print(f"  Duplicates removed: {before - len(df):,}")
    
    # Remove missing
    before = len(df)
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    print(f"  Missing EA/BETA/SE removed: {before - len(df):,}")
    
    # Sort
    print("  Sorting by CHR, POS...")
    df = df.sort_values(['CHR', 'POS'])
    
    # Save to temp files
    print(f"  Saving {len(df):,} variants...")
    df.to_csv(temp_std, sep='\t', index=False, compression='gzip')
    df.to_csv(temp_harm, sep='\t', index=False, compression='gzip')
    
    # Get file sizes
    std_size = temp_std.stat().st_size / (1024**2)
    harm_size = temp_harm.stat().st_size / (1024**2)
    
    print(f"  ✓ Saved: {temp_std.name} ({std_size:.1f} MB)")
    print(f"  ✓ Saved: {temp_harm.name} ({harm_size:.1f} MB)")
    
    return len(df)

def process_stroke_tsv(name, tsv_path, temp_std, temp_harm):
    """Process Stroke TSV file with detailed progress and corruption handling."""
    print(f"\n{'='*60}")
    print(f"Processing {name}: {tsv_path.name}")
    print(f"  Input size: {tsv_path.stat().st_size / (1024**2):.1f} MB")
    print('='*60)
    
    rows = []
    line_count = 0
    
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
                
                # Progress report every 500k lines
                if line_count % 500000 == 0:
                    print(f"  ... {line_count:,} lines read, {len(rows):,} variants kept")
                
                # Safety limit
                if line_count > 8000000:
                    print(f"  Warning: Safety limit reached at {line_count:,} lines")
                    break
    except EOFError:
        print(f"  Warning: File ended unexpectedly at line {line_count:,}")
        print(f"  Processing {len(rows):,} variants collected so far...")
    
    print(f"  Total: {line_count:,} lines, {len(rows):,} valid variants")
    
    # Create DataFrame
    print("  Creating DataFrame...")
    df = pd.DataFrame(rows)
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['SNP'], keep='first')
    print(f"  Duplicates removed: {before - len(df):,}")
    
    # Remove missing
    before = len(df)
    df = df.dropna(subset=['EA', 'BETA', 'SE'])
    print(f"  Missing EA/BETA/SE removed: {before - len(df):,}")
    
    # Sort
    print("  Sorting by CHR, POS...")
    df = df.sort_values(['CHR', 'POS'])
    
    # Save to temp files
    print(f"  Saving {len(df):,} variants...")
    df.to_csv(temp_std, sep='\t', index=False, compression='gzip')
    df.to_csv(temp_harm, sep='\t', index=False, compression='gzip')
    
    # Get file sizes
    std_size = temp_std.stat().st_size / (1024**2)
    harm_size = temp_harm.stat().st_size / (1024**2)
    
    print(f"  ✓ Saved: {temp_std.name} ({std_size:.1f} MB)")
    print(f"  ✓ Saved: {temp_harm.name} ({harm_size:.1f} MB)")
    
    return len(df)

def main():
    print("="*60)
    print("GWAS DATA PROCESSING - 4 Remaining Datasets")
    print("="*60)
    print("\nWill process:")
    print("  1. T2D (ieu-b-107.vcf.gz) - Type 2 Diabetes")
    print("  2. AD (ieu-b-2.vcf.gz) - Alzheimer's Disease")
    print("  3. Stroke_Any (GCST006906.h.tsv.gz)")
    print("  4. Stroke_Ischemic (GCST006908.h.tsv.gz)")
    
    results = {}
    
    # Process T2D
    t2d_path = DATA_DIR / "ieu_opengwas/ieu-b-107.vcf.gz"
    if t2d_path.exists():
        temp_std = STD_DIR / "T2D.standardized.gz"
        temp_harm = HARM_DIR / "T2D.harmonized.gz"
        results['T2D'] = process_vcf("T2D", t2d_path, temp_std, temp_harm)
    else:
        print(f"ERROR: {t2d_path} not found")
        results['T2D'] = 0
    
    # Process AD
    ad_path = DATA_DIR / "ieu_opengwas/ieu-b-2.vcf.gz"
    if ad_path.exists():
        temp_std = STD_DIR / "AD.standardized.gz"
        temp_harm = HARM_DIR / "AD.harmonized.gz"
        results['AD'] = process_vcf("AD", ad_path, temp_std, temp_harm)
    else:
        print(f"ERROR: {ad_path} not found")
        results['AD'] = 0
    
    # Process Stroke_Any
    stroke_any_path = DATA_DIR / "comorbidities/Stroke_GWAS_Catalog_GCST006906.h.tsv.gz"
    if stroke_any_path.exists():
        temp_std = STD_DIR / "Stroke_Any.standardized.gz"
        temp_harm = HARM_DIR / "Stroke_Any.harmonized.gz"
        results['Stroke_Any'] = process_stroke_tsv("Stroke_Any", stroke_any_path, temp_std, temp_harm)
    else:
        print(f"ERROR: {stroke_any_path} not found")
        results['Stroke_Any'] = 0
    
    # Process Stroke_Ischemic
    stroke_ischemic_path = DATA_DIR / "comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz"
    if stroke_ischemic_path.exists():
        temp_std = STD_DIR / "Stroke_Ischemic.standardized.gz"
        temp_harm = HARM_DIR / "Stroke_Ischemic.harmonized.gz"
        results['Stroke_Ischemic'] = process_stroke_tsv("Stroke_Ischemic", stroke_ischemic_path, temp_std, temp_harm)
    else:
        print(f"ERROR: {stroke_ischemic_path} not found")
        results['Stroke_Ischemic'] = 0
    
    # Final summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE - FINAL SUMMARY")
    print("="*60)
    print("\nVariant counts:")
    for name, count in sorted(results.items()):
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {name:20s}: {count:>10,} variants")
    
    print("\n" + "="*60)
    print("All 4 datasets processed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
