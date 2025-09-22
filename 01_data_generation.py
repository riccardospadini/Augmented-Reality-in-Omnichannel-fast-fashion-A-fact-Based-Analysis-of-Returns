"""
Synthetic Dataset Generator for AR Impact on E-commerce Performance with Customer Segments
Thesis: "Augmented Reality in Omnichannel Fast Fashion: A Fact-Based Impact of Return on Customer Satisfaction"
Author: Riccardo Spadini
Date: October 2025

This module generates a synthetic dataset capturing the effect of Augmented Reality (AR) 
on e-commerce conversions, returns, logistics costs, and CO₂ emissions.
It also includes customer segmentation to analyze differential impacts and a Scenario Design analysis
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION BLOCK - editable parameters for personal change in scenario design
# =============================================================================

# Random seed for reproducibility
SEED = 42

# Sample size per scenario
N_USERS_PER_SCENARIO = 100000  # Default sample size for each scenario

# Customer segmentation parameters
SEGMENT_A_PROPORTION = 0.5  # 50% of customers are tech-savvy (Segment A)
SEGMENT_B_PROPORTION = 0.5  # 50% of customers are traditional (Segment B)

# =============================================================================
# SCENARIO-SPECIFIC PARAMETERS WITH CUSTOMER SEGMENTS
# Based on Section 3.3.3 of methodology
# =============================================================================

# PESSIMISTIC SCENARIO - Worst-case without AR
# Reflects businesses with no virtual try-on solution
PESSIMISTIC_PARAMS = {
    # Aggregate parameters (for backward compatibility)
    'conv_rate': 0.020,           # 2% conversion (lower end of observed rates)
    'return_rate': 0.400,          # 40% return rate (fashion extremes)
    'aov_mean': 85.0,             # €85 mean AOV (dissatisfied shoppers spend less)
    'aov_std': 30.0,              # Standard deviation for AOV
    
    # Segment A parameters (tech-savvy users)
    'conv_rate_A': 0.025,         # Slightly higher even without AR
    'return_rate_A': 0.400,       # Similar return rates without AR
    'aov_mean_A': 90.0,           # Slightly higher AOV
    'aov_std_A': 32.0,
    
    # Segment B parameters (traditional users)  
    'conv_rate_B': 0.015,         # Lower conversion
    'return_rate_B': 0.400,       # Similar return rates without AR
    'aov_mean_B': 80.0,           # Lower AOV
    'aov_std_B': 28.0,
    
    'scenario_name': 'pessimistic'
}

# BASE SCENARIO - Current typical situation
# Average industry condition without AR
BASE_PARAMS = {
    # Aggregate parameters
    'conv_rate': 0.030,           # 3% conversion (Toptal 2024, Dynamic Yield)
    'return_rate': 0.325,          # 32.5% return rate (InsideEcology 2024)
    'aov_mean': 100.0,            # €100 mean AOV (OpenSend reports)
    'aov_std': 40.0,              # Standard deviation for AOV
    
    # Segment A parameters (tech-savvy users)
    'conv_rate_A': 0.035,         # ~4% conversion (higher baseline)
    'return_rate_A': 0.325,       # Similar return rates without AR
    'aov_mean_A': 110.0,          # Higher AOV
    'aov_std_A': 45.0,
    
    # Segment B parameters (traditional users)
    'conv_rate_B': 0.025,         # ~2.5% conversion (lower baseline)
    'return_rate_B': 0.325,       # Similar return rates without AR  
    'aov_mean_B': 90.0,           # Lower AOV
    'aov_std_B': 35.0,
    
    'scenario_name': 'base'
}

# OPTIMISTIC SCENARIO - Best-case with AR fully adopted
# AR try-on is fully embraced and effective
OPTIMISTIC_PARAMS = {
    # Aggregate parameters
    'conv_rate': 0.050,           # 5% conversion (~67% uplift from base)
    'return_rate': 0.100,          # 10% return rate (2/3 reduction from base)
    'aov_mean': 120.0,            # €120 mean AOV (20% increase from AR upselling)
    'aov_std': 45.0,              # Standard deviation for AOV
    
    # Segment A parameters (tech-savvy early adopters benefit most)
    'conv_rate_A': 0.060,         # ~6% conversion (major improvement)
    'return_rate_A': 0.050,       # ~5% return rate (dramatic reduction)
    'aov_mean_A': 135.0,          # Higher AOV from AR engagement
    'aov_std_A': 50.0,
    
    # Segment B parameters (traditional users see modest gains)
    'conv_rate_B': 0.040,         # ~4% conversion (moderate improvement)
    'return_rate_B': 0.150,       # ~15% return rate (modest reduction)
    'aov_mean_B': 105.0,          # Smaller AOV increase
    'aov_std_B': 40.0,
    
    'scenario_name': 'optimistic'
}

# CO₂ emission parameters (constant across scenarios)
# Source: CleanHub (2024) - returns add ~30% to delivery emissions
CO2_BASE_PER_ORDER = 1.0         # kg CO₂ for outbound delivery
CO2_RETURN_MULTIPLIER = 1.30     # 30% increase for returned orders
CO2_NOISE_STD = 0.05             # Small noise to avoid degeneracy

# Logistics cost parameters (constant across scenarios)
# Based on industry averages for fashion e-commerce
COST_OUTBOUND = 5.0               # EUR per shipped order
COST_RETURN_SHIPPING = 5.0        # EUR for return shipping
VALUE_LOSS_FRACTION = 0.30        # 30% of AOV lost on returns (Zak, 2024; NRF)
COST_PROCESSING = 2.0             # EUR for handling/restocking

# Unobserved heterogeneity
EPSILON_STD = 0.1                 # Standard deviation for random noise in logit models

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid function for probability transformation.
    
    Parameters
    ----------
    x : np.ndarray
        Input values (logit scale)
    
    Returns
    -------
    np.ndarray
        Probabilities in (0, 1)
    """
    return 1 / (1 + np.exp(-x))


def assign_customer_segments(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Assign customers to segments A (tech-savvy) or B (traditional).
    
    Parameters
    ----------
    n : int
        Number of customers
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Array of segment assignments ('A' or 'B')
    """
    rng = np.random.default_rng(seed)
    segments = rng.choice(['A', 'B'], size=n, 
                         p=[SEGMENT_A_PROPORTION, SEGMENT_B_PROPORTION])
    return segments


def generate_scenario_data(
    n: int,
    scenario_params: dict,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic data for a specific scenario with customer segments.
    
    Parameters
    ----------
    n : int
        Number of users to simulate
    scenario_params : dict
        Dictionary containing scenario-specific parameters
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Dataset for the specified scenario with customer segments
    """
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # Extract scenario parameters
    scenario_name = scenario_params['scenario_name']
    
    # Generate user IDs and assign segments
    user_ids = np.arange(1, n + 1)
    customer_segments = assign_customer_segments(n, seed)
    
    # Initialize arrays
    conversions = np.zeros(n, dtype=int)
    returns = np.zeros(n, dtype=int)
    aov_values = np.zeros(n)
    co2_emissions = np.zeros(n)
    logistics_costs = np.zeros(n)
    
    # ===================
    # SEGMENT-SPECIFIC PROCESSING
    # ===================
    
    for segment in ['A', 'B']:
        segment_mask = customer_segments == segment
        n_segment = segment_mask.sum()
        
        if n_segment == 0:
            continue
            
        # Get segment-specific parameters
        conv_rate = scenario_params[f'conv_rate_{segment}']
        return_rate = scenario_params[f'return_rate_{segment}']
        aov_mean = scenario_params[f'aov_mean_{segment}']
        aov_std = scenario_params[f'aov_std_{segment}']
        
        # ===================
        # CONVERSION MODEL (by segment)
        # ===================
        
        # Calculate base conversion logit
        base_conv_logit = np.log(conv_rate / (1 - conv_rate))
        
        # Build conversion logit with unobserved heterogeneity
        conv_logits = base_conv_logit + rng.normal(0, EPSILON_STD, n_segment)
        
        # Convert to probabilities and sample conversions
        conv_probs = sigmoid(conv_logits)
        conv_probs = np.clip(conv_probs, 0.001, 0.999)
        segment_conversions = rng.binomial(1, conv_probs)
        conversions[segment_mask] = segment_conversions
        
        # ===================
        # AOV MODEL (by segment)
        # ===================
        
        # Only generate AOV for converters in this segment
        segment_converters = segment_conversions == 1
        n_segment_converters = segment_converters.sum()
        
        if n_segment_converters > 0:
            # Sample AOV from normal distribution, ensure positive values
            aov_samples = rng.normal(aov_mean, aov_std, n_segment_converters)
            segment_aov = np.maximum(20.0, aov_samples)  # Minimum €20
            
            # Assign to the full array
            segment_indices = np.where(segment_mask)[0]
            converter_indices = segment_indices[segment_converters]
            aov_values[converter_indices] = segment_aov
        
        # ===================
        # RETURN MODEL (by segment)
        # ===================
        
        if n_segment_converters > 0:
            # Calculate base return logit
            base_return_logit = np.log(return_rate / (1 - return_rate))
            
            # Build return logit with unobserved heterogeneity
            return_logits = base_return_logit + rng.normal(0, EPSILON_STD, n_segment_converters)
            
            # Convert to probabilities and sample returns
            return_probs = sigmoid(return_logits)
            return_probs = np.clip(return_probs, 0.001, 0.999)
            segment_returns = rng.binomial(1, return_probs)
            
            # Assign to the full array
            returns[converter_indices] = segment_returns
    
    # ===================
    # CO₂ EMISSIONS MODEL (same across segments)
    # ===================
    
    # Only orders generate emissions
    order_mask = conversions == 1
    if order_mask.sum() > 0:
        # Base emissions for all orders
        co2_emissions[order_mask] = CO2_BASE_PER_ORDER
        
        # Additional emissions for returns
        return_mask = (returns == 1)
        co2_emissions[return_mask] = CO2_BASE_PER_ORDER * CO2_RETURN_MULTIPLIER
        
        # Add small noise to avoid degeneracy
        co2_emissions[order_mask] += rng.normal(0, CO2_NOISE_STD, order_mask.sum())
        co2_emissions = np.maximum(0, co2_emissions)
    
    # ===================
    # LOGISTICS COST MODEL (same across segments)
    # ===================
    
    if order_mask.sum() > 0:
        # Outbound shipping cost for all orders
        logistics_costs[order_mask] = COST_OUTBOUND
        
        # Additional costs for returns
        return_indices = np.where(returns == 1)[0]
        for idx in return_indices:
            logistics_costs[idx] += COST_RETURN_SHIPPING
            logistics_costs[idx] += VALUE_LOSS_FRACTION * aov_values[idx]
            logistics_costs[idx] += COST_PROCESSING
    
    # ===================
    # CONSTRUCT DATAFRAME
    # ===================
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'scenario': scenario_name,
        'customer_segment': customer_segments,
        'conversion': conversions,
        'return': returns,
        'aov': np.round(aov_values, 2),
        'co2_kg': np.round(co2_emissions, 3),
        'logistic_cost_eur': np.round(logistics_costs, 2)
    })
    
    return df


def generate_full_dataset(
    n_per_scenario: int = N_USERS_PER_SCENARIO,
    seed: int = SEED
) -> pd.DataFrame:
    """
    Generate complete dataset with all three scenarios and customer segments.
    
    This function generates data for pessimistic, base, and optimistic scenarios
    and combines them into a single dataset with customer segment analysis.
    
    Parameters
    ----------
    n_per_scenario : int, optional
        Number of users per scenario (default: N_USERS_PER_SCENARIO)
    seed : int, optional
        Random seed for reproducibility (default: SEED)
    
    Returns
    -------
    pd.DataFrame
        Combined dataset with all scenarios and customer segments
    
    Notes
    -----
    The three scenarios represent:
    1. Pessimistic: Worst-case without AR (high returns, low conversion)
    2. Base: Current typical e-commerce metrics
    3. Optimistic: Best-case with AR fully adopted
    
    Customer segments:
    - Segment A: Tech-savvy early adopters (higher AR benefit)
    - Segment B: Traditional shoppers (modest AR benefit)
    
    References
    ----------
    - Scenario design based on Section 3.3.3 of methodology document
    - Parameters calibrated from multiple industry sources (2024-2025)
    - Segmentation approach from Chen et al. (2024)
    """
    
    # Generate data for each scenario
    scenarios = [PESSIMISTIC_PARAMS, BASE_PARAMS, OPTIMISTIC_PARAMS]
    dfs = []
    
    for i, scenario_params in enumerate(scenarios):
        # Use different seed for each scenario to ensure independence
        scenario_seed = seed + i if seed is not None else None
        
        df_scenario = generate_scenario_data(
            n=n_per_scenario,
            scenario_params=scenario_params,
            seed=scenario_seed
        )
        dfs.append(df_scenario)
    
    # Combine all scenarios
    df_full = pd.concat(dfs, ignore_index=True)
    
    # Reorder columns
    df_full = df_full[['user_id', 'scenario', 'customer_segment', 'conversion', 
                       'return', 'aov', 'co2_kg', 'logistic_cost_eur']]
    
    # ===================
    # VALIDATION CHECKS
    # ===================
    
    # Ensure logical consistency
    assert (df_full.loc[df_full['conversion'] == 0, 'return'] == 0).all(), \
        "Error: Returns detected for non-converters"
    
    assert (df_full.loc[df_full['conversion'] == 0, 'aov'] == 0).all(), \
        "Error: Non-zero AOV for non-converters"
    
    assert (df_full['aov'] >= 0).all(), \
        "Error: Negative AOV values detected"
    
    assert (df_full['co2_kg'] >= 0).all(), \
        "Error: Negative CO₂ emissions detected"
    
    assert (df_full['logistic_cost_eur'] >= 0).all(), \
        "Error: Negative logistics costs detected"
    
    # Check customer segment proportions
    for scenario in ['pessimistic', 'base', 'optimistic']:
        scenario_data = df_full[df_full['scenario'] == scenario]
        seg_a_prop = (scenario_data['customer_segment'] == 'A').mean()
        assert 0.45 <= seg_a_prop <= 0.55, \
            f"Segment A proportion {seg_a_prop:.3f} deviates from expected 50%"
    
    # Check scenario-specific metrics are in expected ranges
    for scenario in ['pessimistic', 'base', 'optimistic']:
        scenario_data = df_full[df_full['scenario'] == scenario]
        conv_rate = scenario_data['conversion'].mean()
        
        if scenario == 'pessimistic':
            assert 0.015 <= conv_rate <= 0.025, \
                f"Pessimistic conversion rate {conv_rate:.3f} out of expected range"
        elif scenario == 'base':
            assert 0.025 <= conv_rate <= 0.035, \
                f"Base conversion rate {conv_rate:.3f} out of expected range"
        elif scenario == 'optimistic':
            assert 0.045 <= conv_rate <= 0.055, \
                f"Optimistic conversion rate {conv_rate:.3f} out of expected range"
    
    return df_full


def compute_scenario_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute key performance indicators by scenario.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset generated by generate_full_dataset()
    
    Returns
    -------
    pd.DataFrame
        KPI summary with metrics for each scenario
    """
    
    kpis = []
    
    for scenario in ['pessimistic', 'base', 'optimistic']:
        scenario_data = df[df['scenario'] == scenario]
        n_users = len(scenario_data)
        n_converters = scenario_data['conversion'].sum()
        
        # Conversion metrics
        conv_rate = scenario_data['conversion'].mean() * 100
        
        # Return metrics (conditional on conversion)
        if n_converters > 0:
            return_rate = scenario_data.loc[
                scenario_data['conversion'] == 1, 'return'
            ].mean() * 100
            aov_buyers = scenario_data.loc[
                scenario_data['conversion'] == 1, 'aov'
            ].mean()
        else:
            return_rate = 0.0
            aov_buyers = 0.0
        
        # Per-user metrics
        avg_cost_per_user = scenario_data['logistic_cost_eur'].mean()
        avg_co2_per_user = scenario_data['co2_kg'].mean()
        
        # Per-order metrics
        if n_converters > 0:
            avg_co2_per_order = scenario_data.loc[
                scenario_data['conversion'] == 1, 'co2_kg'
            ].mean()
        else:
            avg_co2_per_order = 0.0
        
        # Revenue per user (ARPU)
        revenue_per_user = (scenario_data['aov'] * 
                           (1 - scenario_data['return'] * VALUE_LOSS_FRACTION)).mean()
        
        kpis.append({
            'Scenario': scenario.capitalize(),
            'N Users': n_users,
            'Conversion Rate (%)': round(conv_rate, 2),
            'Return Rate on Buyers (%)': round(return_rate, 2),
            'AOV on Buyers (EUR)': round(aov_buyers, 2),
            'Revenue per User (EUR)': round(revenue_per_user, 2),
            'Avg Cost per User (EUR)': round(avg_cost_per_user, 2),
            'Avg CO₂ per User (kg)': round(avg_co2_per_user, 3),
            'Avg CO₂ per Order (kg)': round(avg_co2_per_order, 3)
        })
    
    return pd.DataFrame(kpis)


def compute_segment_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute key performance indicators by scenario and customer segment.
    
    This function provides detailed segment-wise analysis highlighting 
    heterogeneity in AR adoption and impact between tech-savvy and 
    traditional customer groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset generated by generate_full_dataset()
    
    Returns
    -------
    pd.DataFrame
        KPI summary with metrics for each scenario-segment combination
    """
    
    kpis = []
    
    for scenario in ['pessimistic', 'base', 'optimistic']:
        for segment in ['A', 'B']:
            segment_data = df[
                (df['scenario'] == scenario) & 
                (df['customer_segment'] == segment)
            ]
            
            n_users = len(segment_data)
            n_converters = segment_data['conversion'].sum()
            
            # Conversion metrics
            conv_rate = segment_data['conversion'].mean() * 100
            
            # Return metrics (conditional on conversion)
            if n_converters > 0:
                return_rate = segment_data.loc[
                    segment_data['conversion'] == 1, 'return'
                ].mean() * 100
                aov_buyers = segment_data.loc[
                    segment_data['conversion'] == 1, 'aov'
                ].mean()
            else:
                return_rate = 0.0
                aov_buyers = 0.0
            
            # Per-user metrics
            avg_cost_per_user = segment_data['logistic_cost_eur'].mean()
            avg_co2_per_user = segment_data['co2_kg'].mean()
            
            # Revenue per user (ARPU)
            revenue_per_user = (segment_data['aov'] * 
                               (1 - segment_data['return'] * VALUE_LOSS_FRACTION)).mean()
            
            # Segment description
            segment_desc = "Tech-savvy (A)" if segment == 'A' else "Traditional (B)"
            
            kpis.append({
                'Scenario': scenario.capitalize(),
                'Segment': segment_desc,
                'N Users': n_users,
                'Conversion Rate (%)': round(conv_rate, 2),
                'Return Rate on Buyers (%)': round(return_rate, 2),
                'AOV on Buyers (EUR)': round(aov_buyers, 2),
                'Revenue per User (EUR)': round(revenue_per_user, 2),
                'Avg Cost per User (EUR)': round(avg_cost_per_user, 2),
                'Avg CO₂ per User (kg)': round(avg_co2_per_user, 3)
            })
    
    return pd.DataFrame(kpis)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("SYNTHETIC E-COMMERCE AR DATASET GENERATOR WITH CUSTOMER SEGMENTS")
    print("Thesis: AR Impact on Fast Fashion Returns and Sustainability")
    print("=" * 80)
    print()
    
    # Generate complete dataset with all scenarios
    print(f"Generating dataset with {N_USERS_PER_SCENARIO:,} users per scenario...")
    print(f"- Scenarios: Pessimistic, Base, Optimistic")
    print(f"- Customer Segments: A (Tech-savvy), B (Traditional)")
    print(f"- Segment A proportion: {SEGMENT_A_PROPORTION*100}%")
    print(f"- Segment B proportion: {SEGMENT_B_PROPORTION*100}%")
    print(f"- Total users: {N_USERS_PER_SCENARIO * 3:,}")
    print(f"- Random seed: {SEED}")
    print()
    
    df = generate_full_dataset(n_per_scenario=N_USERS_PER_SCENARIO, seed=SEED)
    
    # Save to CSV
    output_filename = "synthetic_ecom_ar_dataset.csv"
    df.to_csv(output_filename, index=False)
    print(f"✓ Dataset saved to: {output_filename}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()
    
    # Compute and display overall KPIs
    print("KEY PERFORMANCE INDICATORS BY SCENARIO")
    print("-" * 80)
    
    kpi_summary = compute_scenario_kpis(df)
    
    # Display detailed KPI comparison
    print("\nAGGREGATE SCENARIO COMPARISON:")
    print("=" * 80)
    
    for _, row in kpi_summary.iterrows():
        scenario_name = row['Scenario']
        
        # Color coding for terminal output (optional)
        if scenario_name == 'Pessimistic':
            print(f"\n📉 {scenario_name} Scenario (No AR, Worst-case):")
        elif scenario_name == 'Base':
            print(f"\n📊 {scenario_name} Scenario (No AR, Industry Average):")
        else:
            print(f"\n📈 {scenario_name} Scenario (With AR, Best-case):")
        
        print(f"  • Users simulated: {row['N Users']:,}")
        print(f"  • Conversion Rate: {row['Conversion Rate (%)']}%")
        print(f"  • Return Rate (on buyers): {row['Return Rate on Buyers (%)']}%")
        print(f"  • AOV (on buyers): €{row['AOV on Buyers (EUR)']}")
        print(f"  • Revenue per User: €{row['Revenue per User (EUR)']}")
        print(f"  • Logistics Cost per User: €{row['Avg Cost per User (EUR)']}")
        print(f"  • CO₂ per User: {row['Avg CO₂ per User (kg)']} kg")
        print(f"  • CO₂ per Order: {row['Avg CO₂ per Order (kg)']} kg")
    
    # Compute and display segment-wise KPIs
    print("\n" + "=" * 80)
    print("SEGMENT-WISE DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    segment_kpis = compute_segment_kpis(df)
    
    for scenario in ['Pessimistic', 'Base', 'Optimistic']:
        print(f"\n {scenario.upper()} SCENARIO - SEGMENT BREAKDOWN:")
        print("-" * 60)
        
        scenario_segments = segment_kpis[segment_kpis['Scenario'] == scenario]
        
        for _, row in scenario_segments.iterrows():
            segment_name = row['Segment']
            print(f"\n  {segment_name}:")
            print(f"    • Conversion Rate: {row['Conversion Rate (%)']}%")
            print(f"    • Return Rate (buyers): {row['Return Rate on Buyers (%)']}%")
            print(f"    • AOV (buyers): €{row['AOV on Buyers (EUR)']}")
            print(f"    • Revenue per User: €{row['Revenue per User (EUR)']}")
            print(f"    • CO₂ per User: {row['Avg CO₂ per User (kg)']} kg")
    
    # Calculate improvements (Optimistic vs Base) by segment
    print("\n" + "=" * 80)
    print("AR IMPACT ANALYSIS BY CUSTOMER SEGMENT (Optimistic vs Base):")
    print("=" * 80)
    
    base_segments = segment_kpis[segment_kpis['Scenario'] == 'Base']
    opt_segments = segment_kpis[segment_kpis['Scenario'] == 'Optimistic']
    
    for segment in ['Tech-savvy (A)', 'Traditional (B)']:
        base_row = base_segments[base_segments['Segment'] == segment].iloc[0]
        opt_row = opt_segments[opt_segments['Segment'] == segment].iloc[0]
        
        conv_improvement = ((opt_row['Conversion Rate (%)'] / 
                            base_row['Conversion Rate (%)']) - 1) * 100
        return_reduction = ((base_row['Return Rate on Buyers (%)'] - 
                            opt_row['Return Rate on Buyers (%)']) / 
                           base_row['Return Rate on Buyers (%)']) * 100
        aov_increase = ((opt_row['AOV on Buyers (EUR)'] / 
                        base_row['AOV on Buyers (EUR)']) - 1) * 100
        co2_reduction = ((base_row['Avg CO₂ per User (kg)'] - 
                         opt_row['Avg CO₂ per User (kg)']) / 
                        base_row['Avg CO₂ per User (kg)']) * 100
        
        print(f"\n {segment} Benefits:")
        print(f"  • Conversion Rate: +{conv_improvement:.1f}% improvement")
        print(f"  • Return Rate: -{return_reduction:.1f}% reduction")
        print(f"  • Average Order Value: +{aov_increase:.1f}% increase")
        print(f"  • CO₂ Emissions: -{co2_reduction:.1f}% reduction per user")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("• Segment A (tech-savvy) shows significantly higher AR adoption benefits")
    print("• Return rate reduction is most pronounced among early adopters")
    print("• Traditional shoppers (Segment B) still benefit but to a lesser extent")
    print("• Heterogeneity supports targeted AR promotion strategies")
    
    print("\n" + "=" * 80)
    print("Dataset generation with customer segments complete!")
    print("=" * 80)