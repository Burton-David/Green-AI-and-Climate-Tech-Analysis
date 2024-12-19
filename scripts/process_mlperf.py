import pandas as pd

def process_mlperf_data():
    """
    Process the MLPerf data with proper tab delimiter handling
    """
    file_path = '../data/Table - Inference_data.csv'
    
    try:
        # Read with explicit tab delimiter and utf-16 encoding
        df = pd.read_csv(file_path, encoding='utf-16', sep='\t')
        
        # Clean up column names
        df.columns = df.columns.str.strip()
        
        print("\nProcessed Dataset Overview:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        
        print("\nColumns after cleanup:")
        for col in df.columns:
            print(f"- {col}")
        
        print("\nUnique Systems:")
        if 'System Name (click + for details)' in df.columns:
            systems = df['System Name (click + for details)'].unique()
            for system in systems[:5]:  # Show first 5
                print(f"- {system}")
                
        print("\nUnique Benchmarks:")
        if 'Benchmark' in df.columns:
            benchmarks = df['Benchmark'].unique()
            for benchmark in benchmarks:
                print(f"- {benchmark}")
        
        print("\nSample Statistics:")
        if 'Avg. Result' in df.columns:
            print(df['Avg. Result'].describe())
        
        # Save processed data
        df.to_csv('../data/mlperf_inference_clean.csv', index=False)
        print("\nSaved cleaned data to 'mlperf_inference_clean.csv'")
        
        return df
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print("Attempting to read raw file content...")
        
        # Read raw file content for debugging
        with open(file_path, 'r', encoding='utf-16') as f:
            first_lines = [next(f) for _ in range(5)]
            print("\nFirst few lines of raw file:")
            for line in first_lines:
                print(line)
        
        return None

if __name__ == "__main__":
    df = process_mlperf_data()