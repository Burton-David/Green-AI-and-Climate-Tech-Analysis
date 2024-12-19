import pandas as pd

def load_mlperf_data():
    """
    Try multiple encodings to load the MLPerf data
    """
    file_path = '../data/Table - Inference_data.csv'
    encodings = ['utf-16', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"\nTrying {encoding} encoding...")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Success! Data loaded with {encoding} encoding")
            
            print("\nDataset Overview:")
            print(f"Number of rows: {len(df)}")
            print(f"Number of columns: {len(df.columns)}")
            
            print("\nColumns:")
            for col in df.columns:
                print(f"- {col}")
            
            print("\nFirst few rows:")
            print(df.head())
            
            # Save with proper encoding
            df.to_csv('../data/mlperf_inference_processed.csv', index=False, encoding='utf-8')
            print("\nSaved processed file as 'mlperf_inference_processed.csv'")
            
            return df
            
        except Exception as e:
            print(f"Failed with {encoding}: {str(e)}")
    
    print("\nFailed to load file with common encodings")
    return None

if __name__ == "__main__":
    df = load_mlperf_data()