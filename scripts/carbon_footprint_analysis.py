import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

class CarbonFootprintAnalyzer:
    def __init__(self):
        self.data_dir = Path('../data')
        self.images_dir = Path('../images')
        self.images_dir.mkdir(exist_ok=True)
        
    def load_and_clean_data(self):
        """Load and clean both datasets"""
        print("Loading and cleaning data...")
        
        # Load eGRID data
        egrid_df = pd.read_excel(self.data_dir / 'egrid2022_data.xlsx', sheet_name='PLNT22')
        
        # Clean eGRID data
        emissions_col = 'Plant annual CO2 total output emission rate (lb/MWh)'
        location_col = 'Plant state abbreviation'
        
        # Remove header row
        egrid_df = egrid_df[egrid_df[location_col] != 'PSTATABB'].copy()
        
        # Convert emissions to numeric
        egrid_df[emissions_col] = pd.to_numeric(egrid_df[emissions_col], errors='coerce')
        
        # Handle outliers
        emissions_data = egrid_df[emissions_col].dropna()
        z_scores = np.abs(stats.zscore(emissions_data))
        valid_mask = z_scores < 3
        egrid_df.loc[emissions_data[~valid_mask].index, emissions_col] = np.nan
        
        print(f"\neGRID Data Summary:")
        print(f"Total plants: {len(egrid_df)}")
        print(f"Plants with valid emission rates: {egrid_df[emissions_col].notna().sum()}")
        print(f"Unique states: {egrid_df[location_col].nunique()}")
        
        # Basic statistics for emissions
        print("\nEmissions Rate Statistics (lb/MWh):")
        print(egrid_df[emissions_col].describe())
        
        # Load MLPerf data
        mlperf_df = pd.read_csv(self.data_dir / 'mlperf_inference_clean.csv')
        
        return egrid_df, mlperf_df
    
    def calculate_regional_carbon_intensity(self, egrid_df):
        """Calculate average carbon intensity by region with proper error handling"""
        print("\nCalculating regional carbon intensity...")
        
        emissions_col = 'Plant annual CO2 total output emission rate (lb/MWh)'
        location_col = 'Plant state abbreviation'
        
        # Create summary statistics by state
        regional_stats = []
        
        for state in egrid_df[location_col].unique():
            state_data = egrid_df[egrid_df[location_col] == state][emissions_col].dropna()
            
            if len(state_data) > 0:
                stats_dict = {
                    'state': state,
                    'mean': state_data.mean(),
                    'std': state_data.std() if len(state_data) > 1 else 0,
                    'count': len(state_data),
                    'median': state_data.median(),
                    'min': state_data.min(),
                    'max': state_data.max()
                }
                
                # Calculate confidence intervals using t-distribution
                if len(state_data) > 1:
                    conf_level = 0.95
                    degrees_of_freedom = len(state_data) - 1
                    t_value = stats.t.ppf((1 + conf_level) / 2, degrees_of_freedom)
                    margin_of_error = t_value * (stats_dict['std'] / np.sqrt(len(state_data)))
                    
                    stats_dict['ci_lower'] = stats_dict['mean'] - margin_of_error
                    stats_dict['ci_upper'] = stats_dict['mean'] + margin_of_error
                else:
                    # For single data points, use the value itself
                    stats_dict['ci_lower'] = stats_dict['mean']
                    stats_dict['ci_upper'] = stats_dict['mean']
                
                regional_stats.append(stats_dict)
        
        regional_intensity = pd.DataFrame(regional_stats)
        
        # Sort by mean emissions rate
        regional_intensity = regional_intensity.sort_values('mean', ascending=False)
        
        print("\nRegional Carbon Intensity Summary:")
        print(f"Number of states analyzed: {len(regional_intensity)}")
        print("\nTop 5 states by average emissions rate (lb/MWh):")
        print(regional_intensity.head()[['state', 'mean', 'count', 'std']].to_string(index=False))
        
        print("\nBottom 5 states by average emissions rate (lb/MWh):")
        print(regional_intensity.tail()[['state', 'mean', 'count', 'std']].to_string(index=False))
        
        return regional_intensity
    
    def analyze_system_power_profiles(self, mlperf_df):
        """Analyze power profiles of different ML systems"""
        print("\nAnalyzing system power profiles...")
        
        # Power consumption estimates (Watts)
        accelerator_power = {
            'H100-SXM': 700,
            'H100-PCIe': 350,
            'A100-SXM': 400,
            'A100-PCIe': 300,
            'L4': 72,
            'T4': 70
        }
        
        # Process system data
        systems = mlperf_df['System Name (click + for details)'].unique()
        system_stats = []
        
        for system in systems:
            system_data = mlperf_df[mlperf_df['System Name (click + for details)'] == system].iloc[0]
            
            # Calculate power consumption
            base_power = 200  # Base system power
            
            # Accelerator power
            acc_power = 0
            if pd.notna(system_data['Accelerator']) and pd.notna(system_data['# of Accelerators']):
                for acc_type, power in accelerator_power.items():
                    if acc_type in str(system_data['Accelerator']):
                        try:
                            acc_count = float(system_data['# of Accelerators'])
                            acc_power = power * acc_count
                        except:
                            pass
                        break
            
            # CPU power
            cpu_power = 150  # Default
            if pd.notna(system_data['Host Processor Core Count']):
                try:
                    cores = float(system_data['Host Processor Core Count'])
                    cpu_power = cores * 5  # 5W per core estimate
                except:
                    pass
            
            total_power = base_power + acc_power + cpu_power
            
            system_stats.append({
                'system': system,
                'accelerator': system_data['Accelerator'],
                'accelerator_count': system_data['# of Accelerators'],
                'processor': system_data['Processor'],
                'core_count': system_data['Host Processor Core Count'],
                'base_power': base_power,
                'acc_power': acc_power,
                'cpu_power': cpu_power,
                'total_power': total_power
            })
        
        system_stats_df = pd.DataFrame(system_stats)
        
        print("\nSystem Power Profile Summary:")
        print(f"Number of unique systems: {len(system_stats_df)}")
        print("\nTop 5 systems by estimated power consumption:")
        top_systems = system_stats_df.nlargest(5, 'total_power')
        print("\n{:<40} {:<10} {:<10} {:<10} {:<10}".format(
            "System", "Total(W)", "ACC(W)", "CPU(W)", "Base(W)"))
        print("-" * 80)
        for _, row in top_systems.iterrows():
            print("{:<40} {:<10.0f} {:<10.0f} {:<10.0f} {:<10.0f}".format(
                row['system'][:39], 
                row['total_power'], 
                row['acc_power'], 
                row['cpu_power'], 
                row['base_power']
            ))
        
        return system_stats_df
    
    def plot_results(self, regional_intensity, system_stats):
        """Create visualizations"""
        print("\nGenerating visualizations...")
        
        # Plot 1: Regional Carbon Intensity
        plt.figure(figsize=(15, 8))
        data = regional_intensity.copy()
        
        plt.errorbar(
            x=range(len(data)),
            y=data['mean'],
            yerr=[
                data['mean'] - data['ci_lower'],
                data['ci_upper'] - data['mean']
            ],
            fmt='o', capsize=5, color='blue', alpha=0.6,
            label='95% Confidence Interval'
        )
        
        plt.xticks(range(len(data)), data['state'], rotation=45, ha='right')
        plt.title('Regional Carbon Intensity by State')
        plt.xlabel('State')
        plt.ylabel('CO2 Emissions Rate (lb/MWh)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.images_dir / 'regional_emissions.png')
        plt.close()
        
        # Plot 2: System Power Consumption
        plt.figure(figsize=(15, 8))
        data = system_stats.nlargest(20, 'total_power')
        
        bottom = np.zeros(len(data))
        
        # Stacked bar chart for power components
        components = ['base_power', 'cpu_power', 'acc_power']
        colors = ['lightgray', 'lightblue', 'darkblue']
        labels = ['Base System', 'CPU', 'Accelerators']
        
        for component, color, label in zip(components, colors, labels):
            plt.bar(data['system'], data[component], bottom=bottom, 
                   label=label, color=color)
            bottom += data[component]
        
        plt.xticks(rotation=45, ha='right')
        plt.title('Estimated Power Consumption by System Component')
        plt.xlabel('System')
        plt.ylabel('Estimated Power (Watts)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.images_dir / 'system_power.png')
        plt.close()
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        # Load and clean data
        egrid_df, mlperf_df = self.load_and_clean_data()
        
        # Calculate regional carbon intensity
        regional_intensity = self.calculate_regional_carbon_intensity(egrid_df)
        
        # Analyze system power profiles
        system_stats = self.analyze_system_power_profiles(mlperf_df)
        
        # Generate visualizations
        self.plot_results(regional_intensity, system_stats)
        
        # Save processed data
        regional_intensity.to_csv(self.data_dir / 'regional_carbon_intensity.csv', index=False)
        system_stats.to_csv(self.data_dir / 'system_power_profiles.csv', index=False)
        
        print("\nAnalysis complete! Check the 'images' directory for visualizations.")
        
        return regional_intensity, system_stats

if __name__ == "__main__":
    analyzer = CarbonFootprintAnalyzer()
    regional_intensity, system_stats = analyzer.run_analysis()