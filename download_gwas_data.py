#!/usr/bin/env python3
"""
Download script for Hypertension Pan-Comorbidity Multi-Modal Atlas GWAS datasets
Step 1A & 1B: Download IEU OpenGWAS and comorbidity datasets
"""

import os
import sys
import json
import time
import gzip
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

try:
    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        Retry = None
except ImportError:
    print("Installing required packages...")
    os.system("pip3 install requests --user")
    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        Retry = None

# Setup paths
BASE_DIR = Path("/Users/yangzi/Desktop/Hypertension Pan-Comorbidity Multi-Modal Atlas")
DATA_DIR = BASE_DIR / "data"
IEU_DIR = DATA_DIR / "ieu_opengwas"
UKB_DIR = DATA_DIR / "ukb_bp"
COMORB_DIR = DATA_DIR / "comorbidities"

# Create directories
for d in [IEU_DIR, UKB_DIR, COMORB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# IEU OpenGWAS datasets configuration
IEU_DATASETS = {
    "SBP_ieu-b-4818.txt.gz": {
        "id": "ieu-b-4818",
        "trait": "Systolic Blood Pressure",
        "output_dir": IEU_DIR
    },
    "DBP_ieu-b-4819.txt.gz": {
        "id": "ieu-b-4819",
        "trait": "Diastolic Blood Pressure",
        "output_dir": IEU_DIR
    },
    "PP_ieu-b-4820.txt.gz": {
        "id": "ieu-b-4820",
        "trait": "Pulse Pressure",
        "output_dir": IEU_DIR
    }
}

# Comorbidity datasets
COMORBIDITY_DATASETS = {
    "Stroke_Any_ieu-b-4424.txt.gz": {"id": "ieu-b-4424", "trait": "Any Stroke"},
    "Stroke_Ischemic_ieu-b-4425.txt.gz": {"id": "ieu-b-4425", "trait": "Ischemic Stroke"},
    "CAD_ieu-b-35.txt.gz": {"id": "ieu-b-35", "trait": "Coronary Artery Disease"},
    "T2D_ieu-b-107.txt.gz": {"id": "ieu-b-107", "trait": "Type 2 Diabetes"},
    "CKD_ieu-b-6049.txt.gz": {"id": "ieu-b-6049", "trait": "Chronic Kidney Disease"},
    "BMI_ieu-a-2.txt.gz": {"id": "ieu-a-2", "trait": "BMI/Obesity"},
    "Alzheimers_ieu-b-2.txt.gz": {"id": "ieu-b-2", "trait": "Alzheimer's Disease"},
    "Depression_ieu-b-102.txt.gz": {"id": "ieu-b-102", "trait": "Depression"}
}

class OpenGWASDownloader:
    """Handler for IEU OpenGWAS downloads"""
    
    def __init__(self, jwt_token=None):
        self.base_url = "https://api.opengwas.io/api"
        self.session = requests.Session()
        
        # Setup retry strategy
        if Retry:
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        
        self.jwt_token = jwt_token
        self.headers = {}
        if jwt_token:
            self.headers["Authorization"] = f"Bearer {jwt_token}"
    
    def download_dataset(self, dataset_id, output_path, max_retries=3):
        """Download a specific IEU OpenGWAS dataset"""
        
        url = f"{self.base_url}/gwasinfo/files"
        
        for attempt in range(max_retries):
            try:
                print(f"  Downloading {dataset_id} (attempt {attempt + 1}/{max_retries})...")
                
                # First, get file info
                response = self.session.post(
                    url,
                    headers={**self.headers, "Content-Type": "application/json"},
                    json={"id": [dataset_id]},
                    timeout=30
                )
                
                if response.status_code == 401:
                    print(f"  ERROR: Authentication required for {dataset_id}")
                    print(f"  Please obtain a JWT token from https://api.opengwas.io/")
                    return False
                
                if response.status_code != 200:
                    print(f"  ERROR: HTTP {response.status_code} for {dataset_id}")
                    return False
                
                data = response.json()
                
                if not data or dataset_id not in data:
                    print(f"  ERROR: No download info found for {dataset_id}")
                    return False
                
                file_info = data[dataset_id]
                
                # Download the file
                if "file" in file_info:
                    download_url = file_info["file"]
                elif "url" in file_info:
                    download_url = file_info["url"]
                else:
                    print(f"  ERROR: No download URL found for {dataset_id}")
                    return False
                
                # Download with streaming
                print(f"  Downloading from: {download_url[:60]}...")
                
                with self.session.get(download_url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    
                    with open(output_path, 'wb') as f:
                        downloaded = 0
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    if downloaded % (1024*1024) < 8192:  # Update every ~1MB
                                        print(f"    Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
                
                print(f"\n  ✓ Successfully downloaded: {output_path}")
                
                # Verify file
                file_size = os.path.getsize(output_path)
                print(f"  File size: {file_size:,} bytes ({file_size/(1024**3):.2f} GB)")
                
                return True
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    return False
        
        return False

def download_ieu_datasets(jwt_token=None):
    """Download all IEU OpenGWAS primary BP datasets"""
    print("=" * 80)
    print("STEP 1A: Downloading IEU OpenGWAS Blood Pressure Datasets")
    print("=" * 80)
    
    downloader = OpenGWASDownloader(jwt_token)
    results = {}
    
    for filename, info in IEU_DATASETS.items():
        output_path = info["output_dir"] / filename
        
        print(f"\n[{info['trait']}] {info['id']}")
        print(f"Output: {output_path}")
        
        if output_path.exists():
            print(f"  File already exists, skipping...")
            results[info['id']] = "exists"
            continue
        
        success = downloader.download_dataset(info['id'], output_path)
        results[info['id']] = "success" if success else "failed"
        
        if not success and not jwt_token:
            print("\n" + "!" * 80)
            print("NOTE: IEU OpenGWAS API requires JWT token for download.")
            print("Please obtain a token from: https://api.opengwas.io/")
            print("Or try downloading manually from: https://opengwas.io/datasets/")
            print("!" * 80)
        
        time.sleep(2)  # Rate limiting
    
    return results

def download_comorbidity_datasets(jwt_token=None):
    """Download all comorbidity trait datasets"""
    print("\n" + "=" * 80)
    print("STEP 1B: Downloading Comorbidity Trait Datasets")
    print("=" * 80)
    
    downloader = OpenGWASDownloader(jwt_token)
    results = {}
    
    for filename, info in COMORBIDITY_DATASETS.items():
        output_path = COMORB_DIR / filename
        
        print(f"\n[{info['trait']}] {info['id']}")
        print(f"Output: {output_path}")
        
        if output_path.exists():
            print(f"  File already exists, skipping...")
            results[info['id']] = "exists"
            continue
        
        success = downloader.download_dataset(info['id'], output_path)
        results[info['id']] = "success" if success else "failed"
        
        time.sleep(2)  # Rate limiting
    
    return results

def download_ukb_bp_dataset():
    """Download UK Biobank BP GWAS from alternative source"""
    print("\n" + "=" * 80)
    print("STEP 1A (continued): Downloading UKB BP Meta-analysis Dataset")
    print("=" * 80)
    
    # Alternative sources for UKB BP data
    ukb_sources = [
        {
            "name": "Figshare Manchester",
            "url": "https://figshare.manchester.ac.uk/articles/dataset/UK_Biobank_blood_pressure_GWAS_summary_statistics_using_337_422_unrelated_white_European_individuals/24851436",
            "files": [
                "https://figshare.manchester.ac.uk/ndownloader/files/43954543",  # SBP
                "https://figshare.manchester.ac.uk/ndownloader/files/43954546",  # DBP
                "https://figshare.manchester.ac.uk/ndownloader/files/43954549"   # PP
            ]
        }
    ]
    
    print("\nUK Biobank BP GWAS sources:")
    print(f"Primary: Evangelou et al. 2018 - 337,422 European ancestry individuals")
    print(f"Alternative download available from Figshare")
    print(f"\nNote: Large files (>2GB total). Manual download recommended from:")
    print(f"  {ukb_sources[0]['url']}")
    print(f"\nOr via command line:")
    print(f"  curl -L -o UKB_BP_SBP.txt.gz {ukb_sources[0]['files'][0]}")
    print(f"  curl -L -o UKB_BP_DBP.txt.gz {ukb_sources[0]['files'][1]}")
    print(f"  curl -L -o UKB_BP_PP.txt.gz {ukb_sources[0]['files'][2]}")
    
    return False

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("HYPERTENSION PAN-COMORBIDITY MULTI-MODAL ATLAS")
    print("GWAS Summary Data Download Script")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {BASE_DIR}")
    print("=" * 80)
    
    # Check for JWT token
    jwt_token = os.environ.get('OPENGWAS_JWT')
    
    if not jwt_token:
        print("\n⚠ WARNING: No OPENGWAS_JWT environment variable found.")
        print("   IEU OpenGWAS API requires authentication.")
        print("   Set JWT token with: export OPENGWAS_JWT='your_token_here'")
        print("   Or download manually from https://opengwas.io/datasets/")
        print("   Attempting download anyway (will likely fail without auth)...\n")
    
    # Step 1A: Download IEU BP datasets
    bp_results = download_ieu_datasets(jwt_token)
    
    # Step 1A: UKB BP dataset info
    ukb_result = download_ukb_bp_dataset()
    
    # Step 1B: Download comorbidity datasets
    comorb_results = download_comorbidity_datasets(jwt_token)
    
    # Summary report
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    
    print("\nPrimary BP Datasets (IEU OpenGWAS):")
    for ds_id, status in bp_results.items():
        icon = "✓" if status == "success" else "⚠" if status == "exists" else "✗"
        print(f"  {icon} {ds_id}: {status}")
    
    print("\nComorbidity Datasets:")
    for ds_id, status in comorb_results.items():
        icon = "✓" if status == "success" else "⚠" if status == "exists" else "✗"
        print(f"  {icon} {ds_id}: {status}")
    
    print("\nUK Biobank BP:")
    print("  ℹ Requires manual download or JWT authentication")
    print("  ℹ Alternative: Download from Figshare (see URLs above)")
    
    print("\n" + "=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check if any downloads succeeded
    all_success = all(s in ["success", "exists"] for s in list(bp_results.values()) + list(comorb_results.values()))
    
    if not jwt_token and not all_success:
        print("\n⚠ AUTHENTICATION REQUIRED")
        print("=" * 80)
        print("To download IEU OpenGWAS datasets, you need to:")
        print("1. Visit https://api.opengwas.io/ and create an account")
        print("2. Generate a JWT token")
        print("3. Run: export OPENGWAS_JWT='your_token_here'")
        print("4. Re-run this script")
        print("\nAlternative: Download files manually from https://opengwas.io/datasets/")
        print("=" * 80)
        return 1
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
