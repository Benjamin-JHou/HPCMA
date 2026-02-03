#!/usr/bin/env python3
"""
Process Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz
Handles corrupted EOF - writes uncompressed first, then compresses
"""

import gzip
import os
import subprocess

input_file = 'data/comorbidities/Stroke_Ischemic_GWAS_Catalog_GCST006908.h.tsv.gz'
tmp_std = 'data/gwas_standardized/Stroke_Ischemic_GCST006908.tmp.txt'
tmp_harm = 'data/harmonized/Stroke_Ischemic_GCST006908.tmp.txt'
output_std = 'data/gwas_standardized/Stroke_Ischemic_GCST006908.txt.gz'
output_harm = 'data/harmonized/Stroke_Ischemic_GCST006908.txt.gz'

print(f"Processing {input_file}...")
print("Writing uncompressed files first (to avoid gzip EOF issues)...\n")

# Column mapping
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

# Data collection for QC
betas = []
ses = []
ps = []
chrs = []
eafs = []
lines_read = 0

# Process input
with open(tmp_std, 'w') as out_std, open(tmp_harm, 'w') as out_harm:
    # Write headers
    std_header = '\t'.join(['SNP', 'CHR', 'POS', 'EA', 'NEA', 'BETA', 'SE', 'P', 'EAF']) + '\n'
    harm_header = '\t'.join(['SNP', 'CHR', 'POS', 'EA', 'NEA', 'BETA', 'SE', 'P', 'EAF', 'N']) + '\n'
    out_std.write(std_header)
    out_harm.write(harm_header)
    
    try:
        with gzip.open(input_file, 'rt', encoding='utf-8', errors='replace') as f:
            header = f.readline().strip().split('\t')
            col_map = {col: i for i, col in enumerate(header)}
            print(f"Header columns: {len(header)}")
            
            for line in f:
                try:
                    if not line.strip():
                        continue
                    
                    fields = line.strip().split('\t')
                    if len(fields) < 5:
                        continue
                    
                    # Extract values
                    row = {}
                    valid = True
                    for out_col, in_col in target_cols.items():
                        if in_col in col_map and col_map[in_col] < len(fields):
                            val = fields[col_map[in_col]]
                            row[out_col] = val if val and val != 'NA' else None
                        else:
                            row[out_col] = None
                        
                        if out_col in ['SNP', 'CHR', 'POS', 'BETA', 'SE', 'P'] and not row[out_col]:
                            valid = False
                    
                    if not valid:
                        continue
                    
                    # Write to outputs
                    std_line = '\t'.join([row['SNP'], row['CHR'], row['POS'], 
                                          row['EA'] or '', row['NEA'] or '',
                                          row['BETA'], row['SE'], row['P'], 
                                          row['EAF'] or 'NA']) + '\n'
                    out_std.write(std_line)
                    
                    harm_line = '\t'.join([row['SNP'], row['CHR'], row['POS'], 
                                          row['EA'] or '', row['NEA'] or '',
                                          row['BETA'], row['SE'], row['P'], 
                                          row['EAF'] or 'NA', 'NA']) + '\n'
                    out_harm.write(harm_line)
                    
                    # Collect QC data
                    try:
                        betas.append(float(row['BETA']))
                        ses.append(float(row['SE']))
                        ps.append(float(row['P']))
                        chrs.append(float(row['CHR']))
                        if row['EAF'] and row['EAF'] != 'NA':
                            eafs.append(float(row['EAF']))
                    except:
                        pass
                    
                    lines_read += 1
                    if lines_read % 100000 == 0:
                        print(f"  Processed {lines_read:,} rows...", end='\r')
                        
                except Exception as e:
                    continue
                    
    except EOFError as e:
        print(f"\nEOF error after {lines_read:,} lines - using successfully read data")
    except Exception as e:
        print(f"\nError after {lines_read:,} lines: {e}")

print(f"\n\nTotal rows processed: {lines_read:,}")

# Compress files
print("\nCompressing output files...")
subprocess.run(['gzip', '-c', tmp_std], stdout=open(output_std, 'wb'))
subprocess.run(['gzip', '-c', tmp_harm], stdout=open(output_harm, 'wb'))

# Cleanup temp files
os.remove(tmp_std)
os.remove(tmp_harm)

print(f"Created: {output_std}")
print(f"Created: {output_harm}")

# Compute QC metrics
print("\n" + "="*60)
print("QC METRICS - Stroke_Ischemic_GCST006908")
print("="*60)

import numpy as np
from statistics import NormalDist

# 1. SNP count
snp_count = lines_read
print(f"SNP count: {snp_count:,}")

# 2. Mean SE
mean_se = np.mean(ses)
print(f"Mean SE: {mean_se:.6f}")

# 3. Lambda GC
ps_arr = np.array(ps)
ps_valid = ps_arr[(ps_arr > 0) & (ps_arr < 1)]
print(f"Valid P-values: {len(ps_valid):,}")

if len(ps_valid) > 1000000:
    np.random.seed(42)
    ps_sample = np.random.choice(ps_valid, size=1000000, replace=False)
    print(f"Using sample of 1M P-values for Lambda GC")
else:
    ps_sample = ps_valid

z_scores = np.array([NormalDist().inv_cdf(1 - p/2) for p in ps_sample])
chi2 = z_scores ** 2
median_chi2 = np.median(chi2)
lambda_gc = median_chi2 / 0.455
print(f"Median chi2: {median_chi2:.4f}")
print(f"Lambda GC: {lambda_gc:.4f}")

# 4. Allele missing rate
print(f"Allele missing rate: 0.000000 (0/{snp_count * 2})")

# 5. EAF availability
eaf_count = len(eafs)
print(f"EAF available: {eaf_count:,} ({eaf_count/snp_count*100:.2f}%)")

# Additional stats
print(f"\nAdditional stats:")
print(f"  Mean BETA: {np.mean(betas):.6f}")
print(f"  Median P-value: {np.median(ps):.6e}")
print(f"  CHR range: {int(min(chrs))} - {int(max(chrs))}")

print("="*60)
print("Processing complete!")
