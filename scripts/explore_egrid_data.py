import pandas as pd
import numpy as np

def explore_egrid_data():
    """
    Explore and clean eGRID data
    """
    print("Loading eGRID data...")
    df = pd.read_excel('../data/egrid2022_data.xlsx', sheet_name='PLNT22')
    
    emissions_col = 'Plant annual CO2 total output emission rate (lb/MWh)'
    location_col = 'Plant state abbreviation'
    
    print("\nChecking emissions column data:")
    print(f"Column name exists: {emissions_col in df.columns}")
    
    if emissions_col in df.columns:
        print("\nSample of emission rate values:")
        print(df[emissions_col].head(10))
        print("\nData type:", df[emissions_col].dtype)
        print("\nUnique values (first 10):")
        print(df[emissions_col].unique()[:10])
        
        # Check for non-numeric values
        non_numeric = df[pd.to_numeric(df[emissions_col], errors='coerce').isna()][emissions_col]
        print("\nNon-numeric values found:")
        print(non_numeric.unique())
    
    print("\nLocation column values:")
    if location_col in df.columns:
        print("\nUnique states:")
        print(df[location_col].unique())
    
    # Try to convert to numeric and get basic stats
    print("\nAttempting to convert to numeric and calculate stats...")
    try:
        numeric_emissions = pd.to_numeric(df[emissions_col], errors='coerce')
        print("\nBasic statistics after conversion:")
        print(numeric_emissions.describe())
        
        # Count nulls
        print("\nNull values:", numeric_emissions.isna().sum())
        
    except Exception as e:
        print(f"Error in conversion: {str(e)}")
    
    return df

if __name__ == "__main__":
    df = explore_egrid_data()