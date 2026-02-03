#!/usr/bin/env python3
"""
Create harmonized file and compute QC metrics for Stroke_Ischemic_GCST006908
"""

import gzip
import pandas as pd
import numpy as np
from statistics import NormalDist

print("Loading standardized file...")
with gzip.open('data/gwas_standardized/Stroke_Ischemic_GCST006908.txt.gz', 'rt') as f:
    std = pd.read_csv(f, sep='\t')

print(f"Loaded {len(std):,} rows")

# Create harmonized file
print("\nCreating harmonized file...")
harm = std.copy()
harm['N'] = np.nan  # Sample size not available in this dataset

# Reorder columns: SNP CHR POS EA NEA BETA SE P EAF N
harm = harm[['SNP', 'CHR', 'POS', 'EA', 'NEA', 'BETA', 'SE', 'P', 'EAF', 'N']]

# Save harmonized
out_harm = 'data/harmonized/Stroke_Ischemic_GCST006908.txt.gz'
print(f"Saving to {out_harm}...")
harm.to_csv(out_harm, sep='\t', index=False, compression='gzip')
print(f"Saved {len(harm):,} rows")

# Compute QC metrics
print("\n" + "="*60)
print("QC METRICS - Stroke_Ischemic_GCST006908")
print("="*60)

# 1. SNP count
snp_count = len(std)
print(f"SNP count: {snp_count:,}")

# 2. Mean SE
mean_se = std['SE'].mean()
print(f"Mean SE: {mean_se:.6f}")

# 3. Lambda GC (median chi2 / 0.455)
print("Computing Lambda GC...")
p_values = std['P'].dropna()
p_values = p_values[(p_values > 0) & (p_values < 1)]
print(f"  Valid P-values: {len(p_values):,}")

# Sample for efficiency
if len(p_values) > 1000000:
    np.random.seed(42)
    p_sample = np.random.choice(p_values, size=1000000, replace=False)
    print(f"  Using sample of 1M P-values")
else:
    p_sample = p_values

# Convert P to Z to chi-square
z_scores = np.array([NormalDist().inv_cdf(1 - p/2) for p in p_sample])
chi2 = z_scores ** 2
median_chi2 = np.median(chi2)
lambda_gc = median_chi2 / 0.455
print(f"  Median chi2: {median_chi2:.4f}")
print(f"Lambda GC: {lambda_gc:.4f}")

# 4. Allele missing rate
total_alleles = len(std) * 2
missing_ea = std['EA'].isna().sum()
missing_nea = std['NEA'].isna().sum()
allele_missing_rate = (missing_ea + missing_nea) / total_alleles
print(f"Allele missing rate: {allele_missing_rate:.6f} ({missing_ea + missing_nea}/{total_alleles})")

# 5. EAF availability
eaf_available = std['EAF'].notna().sum()
print(f"EAF available: {eaf_available:,} ({eaf_available/snp_count*100:.2f}%)")

# Additional summary statistics
print(f"\nAdditional stats:")
print(f"  Mean BETA: {std['BETA'].mean():.6f}")
print(f"  Median P-value: {std['P'].median():.6e}")
print(f"  CHR range: {int(std['CHR'].min())} - {int(std['CHR'].max())}")

print("="*60)
print("Processing complete!")
print(f"\nOutput files:")
print(f"  - data/gwas_standardized/Stroke_Ischemic_GCST006908.txt.gz (already exists)")
print(f"  - {out_harm}")
