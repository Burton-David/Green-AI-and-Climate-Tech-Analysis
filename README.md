# Green AI and Climate Tech Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange.svg)](https://jupyter.org/)

## Overview
Analysis of environmental impact of AI infrastructure, focusing on power consumption patterns, regional variations in carbon impact, and efficiency metrics across different hardware configurations.

## Project Structure
```
green_ai_climate_tech/
├── blog/                # Blog articles
├── data/               # Data files
│   ├── raw/           # Original data files
│   └── processed/     # Cleaned and processed data
├── images/            # Generated visualizations
│   └── analysis/      # Analysis outputs
├── notebooks/         # Jupyter notebooks
├── scripts/          # Analysis scripts
└── tests/            # Test files
```

## Key Features
- Power consumption analysis of AI accelerators
- Geographic impact analysis of data center locations
- Cost-benefit analysis of different hardware configurations
- Visualization of efficiency metrics

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Lab
- Required Python packages (see requirements.txt)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Burton-David/Green-AI-and-Climate-Tech-Analysis.git
cd Green-AI-and-Climate-Tech-Analysis
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage
1. Start Jupyter Lab:
```bash
jupyter lab
```

2. Open the notebooks in the `notebooks/` directory to see the analysis.

## Data Sources
- EPA eGRID data
- MLPerf inference benchmarks

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors
- David Burton - Initial work - [GitHub](https://github.com/Burton-David)

## Acknowledgments
- EPA for providing eGRID data
- MLCommons for benchmark data