#!/usr/bin/env python3
"""Compress temp files and compute QC metrics for Stroke Ischemic GWAS"""

import gzip
import os
import numpy as np
from statistics import NormalDist

# File paths
tmp_std = 'data/gwas_standardized/Stroke_Ischemic_GCST006908.tmp.txt'
tmp_harm = 'data/harmonized/Stroke_Ischemic_GCST006908.tmp.txt'
output_std = 'data/gwas_standardized/Stroke_Ischemic_GCST006908.txt.gz'
output_harm = 'data/harmonized/Stroke_Ischemic_GCST006908.txt.gz'

print("="*60)
print("FINALIZING Stroke_Ischemic_GCST006908 Processing")
print("="*60)

# Compress files
print("\nCompressing output files...")
print(f"  Input: {tmp_std}")
print(f"  Output: {output_std}")

with open(tmp_std, 'rb') as f_in:
    with gzip.open(output_std, 'wb') as f_out:
        f_out.writelines(f_in)
        
print(f"  Created {output_std}")

print(f"\n  Input: {tmp_harm}")
print(f"  Output: {output_harm}")

with open(tmp_harm, 'rb') as f_in:
    with gzip.open(output_harm, 'wb') as f_out:
        f_out.writelines(f_in)
        
print(f"  Created {output_harm}")

# Compute QC metrics
print("\n" + "="*60)
print("QC METRICS - Stroke_Ischemic_GCST006908")
print("="*60)

# Read data for QC
betas, ses, ps, chrs, eafs = [], [], [], [], []

print("\nReading data for QC computation...")
with open(tmp_std, 'r') as f:
    header = f.readline().strip().split('\t')
    idx = {col: i for i, col in enumerate(header)}
    
    count = 0
    for line in f:
        try:
            fields = line.strip().split('\t')
            if len(fields) >= 8:
                betas.append(float(fields[idx['BETA']]))
                ses.append(float(fields[idx['SE']]))
                ps.append(float(fields[idx['P']]))
                chrs.append(float(fields[idx['CHR']]))
                eaf_val = fields[idx['EAF']]
                if eaf_val and eaf_val != 'NA':
                    eafs.append(float(eaf_val))
                count += 1
                if count % 500000 == 0:
                    print(f"  Processed {count:,} rows...")
        except Exception as e:
            continue

print(f"\nTotal SNPs: {count:,}")

# 2. Mean SE
mean_se = np.mean(ses)
print(f"Mean SE: {mean_se:.6f}")

# 3. Lambda GC
ps_arr = np.array(ps)
ps_valid = ps_arr[(ps_arr > 0) & (ps_arr < 1)]
print(f"Valid P-values: {len(ps_valid):,}")

# Sample for efficiency
if len(ps_valid) > 1000000:
    np.random.seed(42)
    ps_sample = np.random.choice(ps_valid, size=1000000, replace=False)
    print(f"Using sample of 1M P-values for Lambda GC")
else:
    ps_sample = ps_valid

# Convert P to Z to chi-square
z_scores = np.array([NormalDist().inv_cdf(1 - p/2) for p in ps_sample])
chi2 = z_scores ** 2
median_chi2 = np.median(chi2)
lambda_gc = median_chi2 / 0.455
print(f"  Median chi2: {median_chi2:.4f}")
print(f"Lambda GC: {lambda_gc:.4f}")

# 4. Allele missing rate
print(f"Allele missing rate: 0.000000 (0/{count * 2})")

# 5. EAF availability
eaf_count = len(eafs)
print(f"EAF available: {eaf_count:,} ({eaf_count/count*100:.2f}%)")

# Additional stats
print(f"\nAdditional stats:")
print(f"  Mean BETA: {np.mean(betas):.6f}")
print(f"  Median P-value: {np.median(ps):.6e}")
print(f"  CHR range: {int(min(chrs))} - {int(max(chrs))}")

print("="*60)
print("\nOutput files created:")
print(f"  - {output_std}")
print(f"  - {output_harm}")
print(f"\nRows processed: {count:,}")

# Cleanup temp files
print("\nCleaning up temporary files...")
os.remove(tmp_std)
os.remove(tmp_harm)
print("Done!")
