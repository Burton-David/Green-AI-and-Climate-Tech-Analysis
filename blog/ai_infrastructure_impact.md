# Environmental Impact Analysis of AI Infrastructure: A Data-Driven Study

*By David Burton*  
*December 19, 2024*

## Introduction

As artificial intelligence continues to advance, understanding its environmental footprint becomes increasingly crucial. This analysis examines the relationship between AI infrastructure choices and their environmental impact, using real-world data from MLPerf benchmarks and EPA power generation statistics.

## Key Findings

- **Power Consumption**: High-end AI accelerators account for up to 88% of system power consumption
- **Regional Impact**: ~5x difference in carbon impact between highest (KS: 2266.76 lb/MWh) and lowest (OR: 431.09 lb/MWh) emission states
- **Hardware Efficiency**: Wide variation in power efficiency across different accelerator types, from 3.04 W/B for H100-PCIe to 138.57 W/B for H200-SXM-141GB-CTS

## Data and Methodology

Our analysis combines two primary datasets:
1. EPA eGRID data covering 11,973 power plants (3,572 with valid emission rates)
2. MLPerf inference benchmarks across 61 unique systems with 15 different accelerator types

### Power Analysis Implementation

Here's how we calculated system power consumption:

```python
def estimate_system_power(row):
    """Estimate total system power consumption"""
    # Base system power
    base_power = 200  # Watts
    
    # CPU power estimation (5W per core)
    cpu_power = float(row['Host Processor Core Count']) * 5 if pd.notna(row['Host Processor Core Count']) else 150
    
    # Accelerator power
    acc_power = 0
    if pd.notna(row['Accelerator']) and pd.notna(row['# of Accelerators']):
        acc_name = str(row['Accelerator'])
        acc_count = float(row['# of Accelerators'])
        
        # Power consumption for different accelerators
        accelerator_power = {
            'H100-SXM': 700,    # SXM variant has higher TDP
            'H100-PCIe': 350,   # PCIe variant is more power-efficient
            'H200-SXM': 700,    # Similar to H100-SXM
            'L40S': 300,        # Professional visualization GPU
            'H100-NVL': 400     # Network variant
        }
        
        for acc_type, power in accelerator_power.items():
            if acc_type in acc_name:
                acc_power = power * acc_count
                break
    
    total_power = base_power + cpu_power + acc_power
    return total_power
```

### Efficiency Metrics

We calculate efficiency in terms of power per billion parameters:

```python
def calculate_efficiency_metrics(df):
    """Calculate efficiency metrics for ML systems"""
    df = df.copy()
    
    # Power per accelerator
    mask = (df['# of Accelerators'] > 0) & (df['accelerator_power'] > 0)
    df.loc[mask, 'watts_per_accelerator'] = (
        df.loc[mask, 'accelerator_power'] / df.loc[mask, '# of Accelerators']
    )
    
    # CO2 impact using KS rate as worst case
    ks_emission_rate = 2266.76
    daily_mwh = (df['total_power'] * 24) / 1_000_000
    df['daily_co2'] = daily_mwh * ks_emission_rate
    
    return df
```

## Results and Analysis

### Power Distribution

Our analysis reveals that accelerators dominate system power consumption. For example, in a typical 8x H100-SXM system:
- Base System: 200W (3.1%)
- CPU: 560W (8.8%)
- Accelerators: 5600W (88.1%)

This power distribution is visualized in our analysis:

[System Power Consumption Breakdown](../images/analysis/power_consumption.png)

### Regional Variations

The carbon impact of AI workloads varies significantly based on location:

```python
Top 5 States by Average Emissions Rate (lb/MWh):
KS: 2266.76
CT: 2211.77
MO: 2024.22
UT: 1994.36
VT: 1938.24

Bottom 5 States by Average Emissions Rate (lb/MWh):
OR: 431.09
NV: 434.92
ID: 446.23
RI: 665.04
CA: 732.03
```

This variation is due to different energy sources in regional power grids, not weather patterns.

### Cost-Benefit Analysis

Our analysis of cost vs. efficiency reveals interesting tradeoffs:

```python
accelerator_costs = {
    'H100-SXM': 30000,
    'H200-SXM': 40000,
    'H100-PCIe': 25000,
    'L40S': 15000,
    'GH200': 35000,
    'MI300X': 25000,
    'TPU': 20000
}
```

[Cost vs Efficiency Tradeoffs](../images/analysis/cost_efficiency_tradeoff.png)

## Conclusions and Recommendations

1. **Hardware Selection**: Consider PCIe variants for better power efficiency when performance requirements allow
2. **Location Impact**: Infrastructure location can have a 5x impact on carbon footprint
3. **System Design**: Focus on accelerator efficiency as it dominates power consumption

## Repository and Code

All code and data used in this analysis are available in our [GitHub repository](https://github.com/Burton-David/Green-AI-and-Climate-Tech-Analysis). The repository includes:
- Data processing scripts
- Analysis notebooks
- Visualization code
- Raw data files

To replicate this analysis:
```bash
git clone https://github.com/Burton-David/Green-AI-and-Climate-Tech-Analysis
cd Green-AI-and-Climate-Tech-Analysis
pip install -r requirements.txt
jupyter lab
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Burton-David/Green-AI-and-Climate-Tech-Analysis/blob/main/LICENSE) file for details.