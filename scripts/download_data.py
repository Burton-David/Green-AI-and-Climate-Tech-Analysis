import os
import requests
import zipfile
import pandas as pd
import hashlib
from pathlib import Path
import logging
import sys
from tqdm import tqdm
import json
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class DataDownloader:
    def __init__(self):
        self.data_dir = Path('../data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url, filename, expected_hash=None):
        """
        Download a file with progress bar and verification
        """
        filepath = self.data_dir / filename
        
        # If file exists and hash matches, skip download
        if filepath.exists() and expected_hash:
            if self._verify_file(filepath, expected_hash):
                logging.info(f"{filename} already exists and hash matches")
                return filepath
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                with open(filepath, 'wb') as f:
                    for data in response.iter_content(block_size):
                        size = f.write(data)
                        pbar.update(size)
            
            if expected_hash and not self._verify_file(filepath, expected_hash):
                raise ValueError(f"Downloaded file {filename} failed hash verification")
                
            logging.info(f"Successfully downloaded {filename}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error downloading {filename}: {str(e)}")
            if filepath.exists():
                filepath.unlink()
            return None

    def _verify_file(self, filepath, expected_hash):
        """
        Verify file integrity using SHA-256
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_hash

    def download_egrid_data(self):
        """
        Download latest eGRID data from EPA
        """
        # EPA eGRID data URL (2022 version)
        url = "https://www.epa.gov/system/files/documents/2024-01/eGRID2022_data.xlsx"
        filepath = self.download_file(url, "egrid2022_data.xlsx")
        
        if filepath:
            try:
                # Convert Excel to CSV for easier handling
                df = pd.read_excel(filepath)
                csv_path = self.data_dir / "egrid2022_data.csv"
                df.to_csv(csv_path, index=False)
                logging.info(f"Converted eGRID data to CSV: {csv_path}")
                return csv_path
            except Exception as e:
                logging.error(f"Error processing eGRID data: {str(e)}")
                return None
        return None

    def download_mlperf_data(self):
        """
        Download MLPerf training results
        """
        # Using GitHub API to get the latest release data
        api_url = "https://api.github.com/repos/mlcommons/training_results_v3.0/contents/NVIDIA/benchmarks/bert/implementations/pytorch-22.09/results"
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            
            results_data = []
            files = response.json()
            
            for file in files:
                if file['name'].endswith('.json'):
                    raw_url = file['download_url']
                    result_response = requests.get(raw_url)
                    result_response.raise_for_status()
                    results_data.append(result_response.json())
            
            # Save combined results
            output_file = self.data_dir / "mlperf_results.json"
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Convert to CSV for easier analysis
            df = self._parse_mlperf_results(results_data)
            csv_path = self.data_dir / "mlperf_results.csv"
            df.to_csv(csv_path, index=False)
            
            logging.info(f"Successfully downloaded and processed MLPerf results")
            return csv_path
            
        except Exception as e:
            logging.error(f"Error downloading MLPerf data: {str(e)}")
            return None

    def _parse_mlperf_results(self, results_data):
        """
        Parse MLPerf JSON results into a pandas DataFrame
        """
        parsed_data = []
        
        for result in results_data:
            try:
                benchmark_data = {
                    'benchmark': result.get('benchmark'),
                    'system_name': result.get('system_name'),
                    'framework': result.get('framework'),
                    'accuracy': result.get('results', [{}])[0].get('accuracy'),
                    'time_to_train': result.get('results', [{}])[0].get('time_to_train'),
                    'epochs': result.get('results', [{}])[0].get('epoch_num')
                }
                parsed_data.append(benchmark_data)
            except Exception as e:
                logging.warning(f"Error parsing result: {str(e)}")
                continue
        
        return pd.DataFrame(parsed_data)

    def verify_downloads(self):
        """
        Verify all required data files exist and are valid
        """
        required_files = [
            "egrid2022_data.csv",
            "mlperf_results.csv",
            "mlperf_results.json"
        ]
        
        missing_files = []
        for file in required_files:
            filepath = self.data_dir / file
            if not filepath.exists():
                missing_files.append(file)
        
        if missing_files:
            logging.warning(f"Missing files: {', '.join(missing_files)}")
            return False
        
        logging.info("All required files present")
        return True

def main():
    downloader = DataDownloader()
    
    # Download eGRID data
    logging.info("Downloading eGRID data...")
    egrid_path = downloader.download_egrid_data()
    if egrid_path:
        logging.info(f"eGRID data downloaded to: {egrid_path}")
    
    # Download MLPerf data
    logging.info("Downloading MLPerf data...")
    mlperf_path = downloader.download_mlperf_data()
    if mlperf_path:
        logging.info(f"MLPerf data downloaded to: {mlperf_path}")
    
    # Verify all downloads
    if downloader.verify_downloads():
        logging.info("All data downloaded successfully!")
    else:
        logging.error("Some data files are missing. Check the log for details.")

if __name__ == "__main__":
    main()
