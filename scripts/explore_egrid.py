import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_egrid_data():
    """
    Explore and summarize the eGRID data
    """
    # Read the Excel file
    data_path = '../data/egrid2022_data.xlsx'
    print(f"\nReading eGRID data from: {data_path}")
    
    try:
        # First, let's list all sheets in the Excel file
        xls = pd.ExcelFile(data_path)
        print("\nAvailable sheets in the Excel file:")
        print(xls.sheet_names)
        
        # Read the first sheet to start
        df = pd.read_excel(data_path, sheet_name=0)
        
        print("\nDataset Overview:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        
        print("\nColumns in the dataset:")
        for col in df.columns:
            print(f"- {col}")
        
        print("\nFirst few rows of data:")
        print(df.head())
        
        # Basic statistics for numeric columns
        print("\nBasic statistics for numeric columns:")
        print(df.describe())
        
        return df
        
    except Exception as e:
        print(f"Error reading the file: {str(e)}")
        return None

if __name__ == "__main__":
    df = explore_egrid_data()
    
    if df is not None:
        print("\nExploration complete. Check the output above for dataset details.")