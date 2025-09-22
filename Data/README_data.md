# Data Description

## Overview
This analysis uses synthetic data generated to simulate e-commerce transactions with AR impact.

## Dataset Structure
- **Size**: 300,000 observations (100,000 users × 3 scenarios)
- **Format**: CSV file (not included due to size - can be regenerated using Code/01_data_generation.py)

## Variables
| Variable | Description | Type | Range |
|----------|-------------|------|-------|
| user_id | Unique user identifier | int | 1-100,000 |
| scenario | Experimental condition | str | pessimistic/base/optimistic |
| customer_segment | User segment | str | A (Early Adopters) / B (Traditional) |
| conversion | Purchase indicator | int | 0/1 |
| return | Return indicator | int | 0/1 |
| aov | Average Order Value | float | 20-250 € |
| logistic_cost_eur | Logistics cost | float | 0-100 € |
| co2_kg | CO2 emissions | float | 0-2 kg |

## Generation Process
Data is synthetically generated based on:
- Industry benchmarks from fast fashion sector
- Academic literature on AR/VTO impact
- Calibrated parameters for realistic scenarios

To regenerate the dataset:
```bash
python Code/01_data_generation.py