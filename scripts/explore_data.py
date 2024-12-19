import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_egrid_data():
    """
    Explore and summarize the eGRID data focusing on plant-level emissions
    """
    data_path = '../data/egrid2022_data.xlsx'
    print(f"\nExploring eGRID data from: {data_path}")
    
    try:
        # Read the PLNT22 sheet (plant-level data)
        df = pd.read_excel(data_path, sheet_name='PLNT22')
        
        print("\neGRID Plant Data Overview:")
        print(f"Number of plants: {len(df)}")
        print(f"Number of metrics: {len(df.columns)}")
        
        print("\nFirst few columns:")
        print(df.columns[:10].tolist())
        
        # Look for emissions-related columns
        emissions_cols = [col for col in df.columns if any(term in col.lower() 
                        for term in ['emission', 'co2', 'carbon', 'ghg'])]
        print("\nEmissions-related columns:")
        for col in emissions_cols:
            print(f"- {col}")
        
        # Basic statistics for key emissions columns
        if emissions_cols:
            print("\nEmissions Statistics:")
            print(df[emissions_cols].describe())
            
        return df
        
    except Exception as e:
        print(f"Error reading eGRID file: {str(e)}")
        return None

def explore_mlperf_data():
    """
    Explore and summarize the MLPerf inference data
    """
    data_path = '../data/Table - Inference_data.csv'
    print(f"\nExploring MLPerf data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        
        print("\nMLPerf Inference Data Overview:")
        print(f"Number of entries: {len(df)}")
        print(f"Number of metrics: {len(df.columns)}")
        
        print("\nColumns in dataset:")
        for col in df.columns:
            print(f"- {col}")
            
        print("\nSample of unique models/systems:")
        if 'System' in df.columns:
            print(df['System'].unique()[:5])
            
        print("\nBasic statistics:")
        print(df.describe())
        
        return df
        
    except Exception as e:
        print(f"Error reading MLPerf file: {str(e)}")
        return None

if __name__ == "__main__":
    # Explore eGRID data
    egrid_df = explore_egrid_data()
    
    # Explore MLPerf data
    mlperf_df = explore_mlperf_data()
    
    if egrid_df is not None and mlperf_df is not None:
        print("\nBoth datasets loaded successfully!")
        
        # Save processed versions if needed
        egrid_df.to_csv('../data/processed_egrid_plant_data.csv', index=False)
        print("Saved processed eGRID data to: processed_egrid_plant_data.csv")
        
        # Output key findings that can help us link the datasets
        print("\nKey Findings:")
        print("-" * 50)
        if egrid_df is not None:
            print(f"Total number of power plants: {len(egrid_df)}")
        if mlperf_df is not None:
            print(f"Total number of ML systems: {len(mlperf_df)}")
