#!/usr/bin/env python3
"""
Quality Control Script for UK Biobank BP GWAS Data
Removes:
- Duplicated rsIDs
- SNPs without effect allele
- SNPs with missing beta or SE

Input: UKB_BP_meta_sumstats.gz or individual trait files
Output: QC-filtered files ready for analysis
"""

import os
import sys
import gzip
import pandas as pd
from pathlib import Path
from datetime import datetime

# Setup paths
BASE_DIR = Path("/Users/yangzi/Desktop/Hypertension Pan-Comorbidity Multi-Modal Atlas")
UKB_DIR = BASE_DIR / "data" / "ukb_bp"
QC_DIR = BASE_DIR / "data" / "ukb_bp_qc"
LOG_DIR = BASE_DIR / "data" / "logs"

# Create directories
QC_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging():
    """Setup logging to file and console"""
    import logging
    log_file = LOG_DIR / f"ukb_qc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def detect_columns(header_line):
    """Detect column names from header"""
    header = header_line.strip().split()
    
    # Common column name mappings
    col_map = {
        'rsid': ['SNP', 'rsID', 'RSID', 'rsid', 'RS_ID', 'RSID', 'MarkerName', 'markername'],
        'chr': ['CHR', 'chromosome', 'Chromosome', 'chr', 'CHROM'],
        'pos': ['POS', 'position', 'Position', 'pos', 'BP'],
        'ea': ['EA', 'A1', 'effect_allele', 'EffectAllele', 'ALT', 'alt', 'Allele1', 'allele1'],
        'nea': ['NEA', 'A2', 'other_allele', 'OtherAllele', 'REF', 'ref', 'Allele2', 'allele2'],
        'beta': ['BETA', 'beta', 'Effect', 'effect', 'OR', 'or', 'LogOdds'],
        'se': ['SE', 'se', 'StdErr', 'stderr', 'standard_error'],
        'pval': ['P', 'p', 'PVALUE', 'pvalue', 'P-value', 'P_val', 'Pval'],
        'eaf': ['EAF', 'eaf', 'FRQ', 'frq', 'Freq', 'freq', 'MAF', 'maf'],
        'n': ['N', 'n', 'sample_size', 'SampleSize']
    }
    
    detected = {}
    for standard, variants in col_map.items():
        for col in header:
            if col in variants:
                detected[standard] = col
                break
    
    return detected, header

def qc_ukb_file(input_file, output_file, logger):
    """
    Apply QC filters to UKB BP GWAS file
    
    Filters:
    1. Remove duplicated rsIDs
    2. Remove SNPs without effect allele
    3. Remove SNPs with missing beta or SE
    4. Remove rows with any NA in critical columns
    """
    
    logger.info(f"Processing: {input_file}")
    logger.info(f"Output: {output_file}")
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return False
    
    # Statistics
    stats = {
        'total_rows': 0,
        'duplicated_rsid': 0,
        'missing_allele': 0,
        'missing_beta': 0,
        'missing_se': 0,
        'final_rows': 0
    }
    
    try:
        # Read the file
        logger.info("Reading input file...")
        
        # Detect compression
        if str(input_file).endswith('.gz'):
            fopen = gzip.open
        else:
            fopen = open
        
        # First pass: detect columns and count total
        with fopen(input_file, 'rt') as f:
            header_line = f.readline()
            col_map, raw_header = detect_columns(header_line)
            
            logger.info(f"Detected columns: {col_map}")
            
            # Check critical columns
            required_cols = ['rsid', 'ea', 'beta', 'se']
            missing_cols = [c for c in required_cols if c not in col_map]
            
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                logger.info(f"Available columns: {raw_header}")
                
                # Try to use available columns
                if 'rsid' not in col_map and 'SNP' in raw_header:
                    col_map['rsid'] = 'SNP'
                if 'ea' not in col_map and 'A1' in raw_header:
                    col_map['ea'] = 'A1'
                if 'beta' not in col_map and 'BETA' in raw_header:
                    col_map['beta'] = 'BETA'
                if 'se' not in col_map and 'SE' in raw_header:
                    col_map['se'] = 'SE'
            
            # Read data in chunks to handle large files
            chunk_size = 100000
            all_data = []
            
            for i, line in enumerate(f):
                stats['total_rows'] += 1
                
                if i % 1000000 == 0 and i > 0:
                    logger.info(f"  Processed {i:,} rows...")
                
                # Stop after reading enough for testing (remove in production)
                if i > 5000000:  # Read first 5M rows for testing
                    logger.info("  Stopping at 5M rows for testing")
                    break
        
        # Use pandas for efficient processing
        logger.info("Loading data with pandas...")
        
        # Determine separator
        if '\t' in header_line:
            sep = '\t'
        elif ',' in header_line:
            sep = ','
        else:
            sep = '\s+'
        
        # Read with pandas
        df = pd.read_csv(
            input_file,
            sep=sep,
            compression='gzip' if str(input_file).endswith('.gz') else None,
            low_memory=False,
            nrows=5000000  # Limit for testing - remove for full processing
        )
        
        initial_rows = len(df)
        logger.info(f"Loaded {initial_rows:,} rows")
        
        # Rename columns to standard names
        reverse_map = {v: k for k, v in col_map.items()}
        df = df.rename(columns=reverse_map)
        
        # Apply QC filters
        logger.info("Applying QC filters...")
        
        # 1. Remove SNPs without effect allele
        if 'ea' in df.columns:
            before = len(df)
            df = df[df['ea'].notna()]
            df = df[df['ea'] != '']
            df = df[df['ea'] != '.']
            stats['missing_allele'] = before - len(df)
            logger.info(f"  Removed {stats['missing_allele']:,} SNPs without effect allele")
        
        # 2. Remove SNPs with missing beta
        if 'beta' in df.columns:
            before = len(df)
            df = df[df['beta'].notna()]
            df = df[df['beta'] != '']
            # Handle common NA values
            df = df[~df['beta'].astype(str).isin(['.', 'NA', 'N/A', 'nan', 'NaN'])]
            stats['missing_beta'] = before - len(df)
            logger.info(f"  Removed {stats['missing_beta']:,} SNPs with missing beta")
        
        # 3. Remove SNPs with missing SE
        if 'se' in df.columns:
            before = len(df)
            df = df[df['se'].notna()]
            df = df[df['se'] != '']
            df = df[~df['se'].astype(str).isin(['.', 'NA', 'N/A', 'nan', 'NaN'])]
            stats['missing_se'] = before - len(df)
            logger.info(f"  Removed {stats['missing_se']:,} SNPs with missing SE")
        
        # 4. Remove duplicated rsIDs (keep first occurrence)
        if 'rsid' in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=['rsid'], keep='first')
            stats['duplicated_rsid'] = before - len(df)
            logger.info(f"  Removed {stats['duplicated_rsid']:,} duplicated rsIDs")
        
        # Convert numeric columns
        for col in ['beta', 'se', 'pval', 'eaf', 'n']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN in critical columns after conversion
        critical_cols = [c for c in ['beta', 'se'] if c in df.columns]
        if critical_cols:
            before = len(df)
            df = df.dropna(subset=critical_cols)
            removed = before - len(df)
            if removed > 0:
                logger.info(f"  Removed {removed:,} rows with NA in critical columns")
        
        stats['final_rows'] = len(df)
        
        # Write output
        logger.info("Writing QC-filtered file...")
        df.to_csv(
            output_file,
            sep='\t',
            compression='gzip',
            index=False
        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("QC SUMMARY")
        logger.info("="*60)
        logger.info(f"Initial rows:     {initial_rows:,}")
        logger.info(f"Final rows:       {stats['final_rows']:,}")
        logger.info(f"Removed total:    {initial_rows - stats['final_rows']:,}")
        logger.info(f"  - Duplicated:   {stats['duplicated_rsid']:,}")
        logger.info(f"  - Missing EA:   {stats['missing_allele']:,}")
        logger.info(f"  - Missing beta: {stats['missing_beta']:,}")
        logger.info(f"  - Missing SE:   {stats['missing_se']:,}")
        logger.info(f"Retention rate:   {stats['final_rows']/initial_rows*100:.1f}%")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main QC workflow"""
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("UKB BP GWAS QC PIPELINE")
    logger.info("="*60)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Look for UKB files
    ukb_files = []
    
    # Check for merged file
    merged_file = UKB_DIR / "UKB_BP_meta_sumstats.gz"
    if merged_file.exists():
        ukb_files.append(('merged', merged_file))
    
    # Check for individual trait files
    trait_files = {
        'SBP': UKB_DIR / "UKB_BP_SBP.txt.gz",
        'DBP': UKB_DIR / "UKB_BP_DBP.txt.gz",
        'PP': UKB_DIR / "UKB_BP_PP.txt.gz"
    }
    
    for trait, fpath in trait_files.items():
        if fpath.exists():
            ukb_files.append((trait, fpath))
    
    if not ukb_files:
        logger.warning("No UKB BP files found in data/ukb_bp/")
        logger.info("Expected files:")
        logger.info("  - UKB_BP_meta_sumstats.gz")
        logger.info("  - UKB_BP_SBP.txt.gz")
        logger.info("  - UKB_BP_DBP.txt.gz")
        logger.info("  - UKB_BP_PP.txt.gz")
        logger.info("\nPlease download UKB BP data first (see DATA_DOWNLOAD_GUIDE.md)")
        return 1
    
    logger.info(f"\nFound {len(ukb_files)} file(s) to process:\n")
    for trait, fpath in ukb_files:
        logger.info(f"  [{trait}] {fpath.name}")
    
    # Process each file
    results = {}
    for trait, input_file in ukb_files:
        logger.info(f"\n{'='*60}")
        output_file = QC_DIR / f"{input_file.stem}_qc.txt.gz"
        success = qc_ukb_file(input_file, output_file, logger)
        results[trait] = success
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    
    for trait, success in results.items():
        status = "✓ QC complete" if success else "✗ Failed"
        logger.info(f"  {trait}: {status}")
    
    logger.info(f"\nQC output directory: {QC_DIR}")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
