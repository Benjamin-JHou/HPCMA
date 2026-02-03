#!/usr/bin/env python3
"""
Process Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz
Handles corrupted EOF error gracefully - streaming approach
"""

import gzip
import pandas as pd
import numpy as np
from statistics import NormalDist
import io

input_file = 'data/comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz'
output_std = 'data/gwas_standardized/Stroke_Ischemic_GCST006908.txt.gz'
output_harm = 'data/harmonized/Stroke_Ischemic_GCST006908.txt.gz'

print(f"Processing {input_file}...")
print("Using streaming approach to handle corrupted EOF...\n")

# Read file with error handling
lines_read = 0
header = None
data_rows = []
chunk_size = 50000
chunks_processed = 0

try:
    with gzip.open(input_file, 'rt', encoding='utf-8', errors='replace') as f:
        # Read header
        header_line = f.readline()
        header = header_line.strip().split('\t')
        print(f"Header columns: {len(header)}")
        
        # Get column indices
        col_map = {col: i for i, col in enumerate(header)}
        target_cols = {
            'SNP': 'hm_rsid',
            'CHR': 'hm_chrom',
            'POS': 'hm_pos',
            'EA': 'hm_effect_allele',
            'NEA': 'hm_other_allele',
            'BETA': 'hm_beta',
            'SE': 'standard_error',
            'P': 'p_value',
            'EAF': 'hm_effect_allele_frequency'
        }
        
        # Verify columns exist
        for out_col, in_col in target_cols.items():
            if in_col not in col_map:
                print(f"Warning: Column {in_col} not found!")
        
        # Process lines
        for line in f:
            try:
                if not line.strip():
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < len(header):
                    continue
                
                # Extract data
                row = {}
                valid = True
                for out_col, in_col in target_cols.items():
                    if in_col in col_map:
                        idx = col_map[in_col]
                        if idx < len(fields):
                            val = fields[idx]
                            row[out_col] = val if val and val != 'NA' else None
                        else:
                            row[out_col] = None
                            if out_col in ['SNP', 'CHR', 'POS', 'BETA', 'SE', 'P']:
                                valid = False
                    else:
                        row[out_col] = None
                        if out_col in ['SNP', 'CHR', 'POS', 'BETA', 'SE', 'P']:
                            valid = False
                
                if valid:
                    data_rows.append(row)
                    lines_read += 1
                    
                    if lines_read % 100000 == 0:
                        print(f"  Lines processed: {lines_read:,}", end='\r')
                        
            except Exception as e:
                # Handle line-level errors
                continue
                
except EOFError as e:
    print(f"\nEOF error encountered after {lines_read:,} lines: {e}")
    print("Continuing with successfully read data...")
except Exception as e:
    print(f"\nError encountered: {e}")
    print(f"Successfully read {lines_read:,} lines before error")

print(f"\n\nTotal valid rows read: {lines_read:,}")

if lines_read == 0:
    print("ERROR: No valid data read!")
    exit(1)

# Create DataFrame
print("\nCreating DataFrame...")
df = pd.DataFrame(data_rows)

# Convert numeric columns
numeric_cols = ['CHR', 'POS', 'BETA', 'SE', 'P', 'EAF']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing critical values
print(f"Rows before cleaning: {len(df):,}")
df_clean = df.dropna(subset=['SNP', 'CHR', 'POS', 'BETA', 'SE', 'P'])
print(f"Rows after cleaning: {len(df_clean):,}")

# Save standardized file
print(f"\nSaving standardized file to {output_std}...")
df_clean.to_csv(output_std, sep='\t', index=False, compression='gzip')
print(f"Saved {len(df_clean):,} rows")

# Create and save harmonized file
print(f"\nCreating harmonized file...")
harm = df_clean.copy()
harm['N'] = np.nan
harm = harm[['SNP', 'CHR', 'POS', 'EA', 'NEA', 'BETA', 'SE', 'P', 'EAF', 'N']]

print(f"Saving harmonized file to {output_harm}...")
harm.to_csv(output_harm, sep='\t', index=False, compression='gzip')
print(f"Saved {len(harm):,} rows")

# Compute QC metrics
print("\n" + "="*60)
print("QC METRICS - Stroke_Ischemic_GCST006908")
print("="*60)

# 1. SNP count
snp_count = len(df_clean)
print(f"SNP count: {snp_count:,}")

# 2. Mean SE
mean_se = df_clean['SE'].mean()
print(f"Mean SE: {mean_se:.6f}")

# 3. Lambda GC (median chi2 / 0.455)
print("Computing Lambda GC...")
p_values = df_clean['P'].dropna()
p_values = p_values[(p_values > 0) & (p_values < 1)]
print(f"  Valid P-values: {len(p_values):,}")

# Sample for efficiency
if len(p_values) > 1000000:
    np.random.seed(42)
    p_sample = np.random.choice(p_values, size=1000000, replace=False)
    print(f"  Using sample of 1M P-values for computation")
else:
    p_sample = p_values.values

# Convert P to Z to chi-square
z_scores = np.array([NormalDist().inv_cdf(1 - p/2) for p in p_sample])
chi2 = z_scores ** 2
median_chi2 = np.median(chi2)
lambda_gc = median_chi2 / 0.455
print(f"  Median chi2: {median_chi2:.4f}")
print(f"Lambda GC: {lambda_gc:.4f}")

# 4. Allele missing rate
total_alleles = len(df_clean) * 2
missing_ea = df_clean['EA'].isna().sum()
missing_nea = df_clean['NEA'].isna().sum()
allele_missing_rate = (missing_ea + missing_nea) / total_alleles
print(f"Allele missing rate: {allele_missing_rate:.6f} ({missing_ea + missing_nea}/{total_alleles})")

# 5. EAF availability
eaf_available = df_clean['EAF'].notna().sum()
print(f"EAF available: {eaf_available:,} ({eaf_available/snp_count*100:.2f}%)")

# Additional summary statistics
print(f"\nAdditional stats:")
print(f"  Mean BETA: {df_clean['BETA'].mean():.6f}")
print(f"  Median P-value: {df_clean['P'].median():.6e}")
print(f"  CHR range: {int(df_clean['CHR'].min())} - {int(df_clean['CHR'].max())}")

print("="*60)
print("Processing complete!")
print(f"\nOutput files:")
print(f"  - {output_std}")
print(f"  - {output_harm}")
