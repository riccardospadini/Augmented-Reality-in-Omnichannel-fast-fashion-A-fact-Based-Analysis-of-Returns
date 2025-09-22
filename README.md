# Augmented Reality in Omnichannel Fast Fashion: A Fact-Based Analysis of Returns on Customer Satisfaction

## Research Overview

This repository contains the code and methodology for analyzing the impact of Augmented Reality (AR) / Virtual Try-On (VTO) technology on customer behavior and returns in fast fashion e-commerce.

**Author**: Riccardo Spadini   
**Institution**: LUISS Guido Carli University  
**Program**: Data Science Management  
**Year**: 2025  

## Research Objectives

1. Quantify the impact of AR on conversion rates and return rates
2. Analyze heterogeneous effects across customer segments
3. Evaluate economic and environmental implications
4. Provide strategic recommendations for AR implementation

## Key Findings

- **Conversion Rate**: +71% increase with AR implementation
- **Return Rate**: -74% reduction for buyers using AR
- **Revenue Impact**: €3.54 additional revenue per user
- **CO₂ Reduction**: 34 tons annually per 1M users
  
## Repository Structure

├── Code/           # Python scripts for analysis
├── Data/           # Data descriptions (synthetic data used)
├── requirements.txt # Package dependencies
└── README.md       # This file

## Installation
# Clone repository
git clone https://github.com/riccardospadini/Augmented-Reality-in-Omnichannel-fast-fashion-A-fact-Based-Analysis-of-Returns.git
cd Augmented-Reality-in-Omnichannel-fast-fashion-A-fact-Based-Analysis-of-Returns

# Install dependencies
pip install -r requirements.txt

## Usage
Run the analysis pipeline in sequence:
bashpython Code/01_data_generation.py      # Generate synthetic dataset
python Code/02_descriptive_analysis.py # Descriptive statistics
python Code/03_regression_analysis.py  # Regression models
python Code/04_monte_carlo_simulation.py # Monte Carlo simulations
python Code/05_visualization.py        # Generate figures

## Methodology (quick summary)

300,000 synthetic observations (100k users × 3 scenarios)
Customer segmentation: Early Adopters (50%) vs Traditional (50%)
Scenarios: Pessimistic, Base, Optimistic
Statistical Analysis: logistic regression (conversion/returns), OLS (economics), Monte Carlo (robustness), heterogeneous effects by segment.

## Citation
If you use this code for research, please cite:
@mastersthesis{spadini2025ar,
  title={Augmented Reality in Omnichannel Fast Fashion: A Fact-Based Analysis of Returns},
  author={[Riccardo Spadini]},
  year={2025},
  school={LUISS Guido Carli University},
  type={Master's Thesis}
}

## License
This project is licensed under the MIT License - see LICENSE file for details.

## Contact 
For questions or collaborations, please contact:
Email: riccardo.spadini@studenti.luiss.it


