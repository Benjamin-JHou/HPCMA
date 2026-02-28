# Hypertension Pan-Comorbidity Multi-Modal Atlas - Data Download Guide

## Step 1A: Primary Blood Pressure Datasets

### Dataset 1: IEU OpenGWAS Blood Pressure GWAS
**Required Output Files:**
- `SBP_ieu-b-4818.txt.gz` (Systolic Blood Pressure)
- `DBP_ieu-b-4819.txt.gz` (Diastolic Blood Pressure)
- `PP_ieu-b-4820.txt.gz` (Pulse Pressure)

**Download Methods:**

#### Method 1: IEU OpenGWAS Website (Manual)
1. Visit https://opengwas.io/datasets/
2. Search for each dataset ID (ieu-b-4818, ieu-b-4819, ieu-b-4820)
3. Click "Download VCF" button
4. Move downloaded files to `data/ieu_opengwas/` with correct names

#### Method 2: IEU OpenGWAS API (Programmatic)
```bash
# 1. Get JWT token from https://api.opengwas.io/
# 2. Set environment variable
export OPENGWAS_JWT='your_token_here'

# 3. Run download script
./download_alternative.sh
```

#### Method 3: R with ieugwasr
```r
# Install package
install.packages('ieugwasr', repos=c('https://mrcieu.r-universe.dev', 'https://cloud.r-project.org'))

# Login to get token
ieugwasr::get_access_token()

# Download files
library(ieugwasr)
datasets <- c("ieu-b-4818", "ieu-b-4819", "ieu-b-4820")
for(ds in datasets) {
  files <- gwasinfo_files(ds)
  download.file(files[[ds]]$file, destfile=paste0(ds, ".txt.gz"))
}
```

### Dataset 2: UK Biobank BP GWAS (Evangelou et al. 2018)
**Dataset Information:**
- **Study:** Evangelou et al. 2018, Nature Genetics
- **Sample:** 337,422 unrelated European ancestry individuals
- **PMID:** 30224653
- **GWAS Catalog:** GCST006624, GCST006625, GCST006626

**Required Output:** `UKB_BP_meta_sumstats.gz`

**Download Sources:**

#### Option 1: Figshare Manchester (Primary)
URL: https://figshare.manchester.ac.uk/articles/dataset/UK_Biobank_blood_pressure_GWAS_summary_statistics_using_337_422_unrelated_white_European_individuals/24851436

Files:
- SBP: https://figshare.manchester.ac.uk/ndownloader/files/43954543
- DBP: https://figshare.manchester.ac.uk/ndownloader/files/43954546
- PP: https://figshare.manchester.ac.uk/ndownloader/files/43954549

Note: May require browser download due to authentication.

#### Option 2: GWAS Catalog (Alternative)
FTP: ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/

Studies:
- GCST006624 - Systolic blood pressure
- GCST006625 - Diastolic blood pressure  
- GCST006626 - Pulse pressure

#### Option 3: Article Supplementary Data
URL: https://www.nature.com/articles/s41588-018-0205-x
Supplementary tables may contain summary statistics links.

#### Option 4: University Repositories
- Glasgow ePrints: https://eprints.gla.ac.uk/169952/
- Check institutional repositories for data availability

**Quality Control Required:**
After downloading, run the QC script to:
- Remove duplicated rsIDs
- Remove SNPs without effect allele
- Remove SNPs with missing beta or SE

## Step 1B: Comorbidity Trait Datasets

### Required Datasets:
1. **Stroke - Any** (ieu-b-4424) → `Stroke_Any_ieu-b-4424.txt.gz`
2. **Stroke - Ischemic** (ieu-b-4425) → `Stroke_Ischemic_ieu-b-4425.txt.gz`
3. **Coronary Artery Disease** (ieu-b-35) → `CAD_ieu-b-35.txt.gz`
4. **Type 2 Diabetes** (ieu-b-107) → `T2D_ieu-b-107.txt.gz`
5. **Chronic Kidney Disease** (ieu-b-6049) → `CKD_ieu-b-6049.txt.gz`
6. **BMI / Obesity** (ieu-a-2) → `BMI_ieu-a-2.txt.gz`
7. **Alzheimer's Disease** (ieu-b-2) → `Alzheimers_ieu-b-2.txt.gz`
8. **Depression** (ieu-b-102) → `Depression_ieu-b-102.txt.gz`

**Download Methods:**
Same as IEU OpenGWAS primary datasets above.

## Alternative Data Sources

### GWAS Catalog
- URL: https://www.ebi.ac.uk/gwas/
- FTP: ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/
- REST API: https://www.ebi.ac.uk/gwas/summary-statistics/api/

### dbGaP
- URL: https://www.ncbi.nlm.nih.gov/gap/
- Some datasets require authorized access

### FinnGen
- URL: https://www.finngen.fi/en/access_results
- GWAS summary statistics for Finnish population

### UK Biobank Neale Lab
- URL: http://www.nealelab.is/uk-biobank
- Round 2 GWAS results (2018)

## Quick Start Commands

```bash
# 1. Set up environment
export OPENGWAS_JWT='your_jwt_token'

# 2. Create directories
mkdir -p data/ieu_opengwas data/ukb_bp data/comorbidities

# 3. Run download script
./download_alternative.sh

# 4. If IEU downloads fail, download manually from:
#    https://opengwas.io/datasets/

# 5. Download UKB BP from Figshare (may need browser):
#    https://figshare.manchester.ac.uk/...

# 6. Apply QC to UKB data
python qc_ukb_data.py
```

## File Naming Convention

All downloaded files should follow this pattern:
```
<trait_description>_<datasetID>.txt.gz
```

Examples:
- `SBP_ieu-b-4818.txt.gz`
- `CAD_ieu-b-35.txt.gz`
- `UKB_BP_meta_sumstats.gz`

## Troubleshooting

### Issue: IEU API returns 401 Unauthorized
**Solution:** You need a valid JWT token
1. Visit https://api.opengwas.io/
2. Create account or login with Google
3. Generate JWT token
4. Set `export OPENGWAS_JWT='token'`

### Issue: Figshare downloads return 0 bytes
**Solution:** Figshare requires browser cookies. Download manually using web browser.

### Issue: Large file downloads fail
**Solution:** Use wget with resume capability:
```bash
wget -c --tries=0 --read-timeout=60 [URL] -O [output]
```

### Issue: API rate limits
**Solution:** IEU OpenGWAS limits to 20 datasets per 24 hours. Wait and retry.

## Expected File Sizes

Approximate sizes for reference:
- IEU BP datasets: ~200-500 MB each (VCF.gz format)
- UKB BP datasets: ~700 MB each (text.gz format)
- Comorbidity datasets: ~100-300 MB each

Total expected: ~3-5 GB

## Data Format

IEU OpenGWAS VCF format contains:
- Chromosome, Position
- rsID
- Reference allele, Alternative allele
- Effect allele frequency
- Beta, Standard Error
- P-value
- Sample size

## Citation Requirements

When using these datasets, please cite:
- **IEU OpenGWAS:** Elsworth et al. (2020) medRxiv
- **UKB BP:** Evangelou et al. (2018) Nature Genetics
- **GWAS Catalog:** Buniello et al. (2019) Nucleic Acids Research

## Contact & Support

- IEU OpenGWAS: https://github.com/MRCIEU/ieugwasr/issues
- GWAS Catalog: gwas-info@ebi.ac.uk
- Figshare: support@figshare.com

## Last Updated
2026-02-28