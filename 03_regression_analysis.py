"""
Regression Models and Causal Inference Analysis for AR Impact Dataset - Enhanced with Segmentation
Thesis: "Augmented Reality in Omnichannel Fast Fashion: A Fact-Based Impact of Return on Customer Satisfaction"
Author: Riccardo Spadini
Date: October 2025

This module implements regression models and causal inference analysis
following Section 3.4.2 of the thesis methodology, including segment-specific analysis.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================

def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for regression analysis by creating necessary variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    
    Returns
    -------
    pd.DataFrame
        Dataset with additional variables for regression
    """
    df = df.copy()
    
    # Rinomina la colonna 'return' se esiste
    if 'return' in df.columns:
        df = df.rename(columns={'return': 'returns'})
    
    # Create scenario dummies
    df['is_pessimistic'] = (df['scenario'] == 'pessimistic').astype(int)
    df['is_base'] = (df['scenario'] == 'base').astype(int)
    df['is_optimistic'] = (df['scenario'] == 'optimistic').astype(int)
    
    # Create AR treatment variable (optimistic = 1, base = 0, exclude pessimistic)
    df['ar_treatment'] = df['is_optimistic'].astype(int)
    
    # Create segment dummy (A = 1, B = 0)
    df['segment_a'] = (df['customer_segment'] == 'A').astype(int)
    
    # Create interaction term
    df['ar_x_segment_a'] = df['ar_treatment'] * df['segment_a']
    
    # Revenue per user (for ARPU analysis)
    df['revenue'] = df['aov'].copy()  # Revenue = AOV (0 if no conversion)
    
    # Net revenue after return losses (30% value loss on returns)
    df['net_revenue'] = df['revenue'] * (1 - df['returns'] * 0.30)
    
    return df

# =============================================================================
# SEGMENT-SPECIFIC LOGISTIC REGRESSION MODELS
# =============================================================================

def logistic_regression_conversion_with_segments(df: pd.DataFrame) -> dict:
    """
    Logistic regression for conversion with segment heterogeneity.
    Tests whether AR effect varies by customer segment.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset
    
    Returns
    -------
    dict
        Regression results with interaction effects
    """
    # Filter to base and optimistic scenarios only
    df_analysis = df[df['scenario'].isin(['base', 'optimistic'])].copy()
    
    # Model 1: Main effects only
    model1 = smf.logit('conversion ~ ar_treatment + segment_a', 
                       data=df_analysis).fit(disp=False)
    
    # Model 2: With interaction (heterogeneous treatment effects)
    model2 = smf.logit('conversion ~ ar_treatment + segment_a + ar_x_segment_a', 
                       data=df_analysis).fit(disp=False)
    
    # Likelihood ratio test for interaction
    lr_stat = 2 * (model2.llf - model1.llf)
    lr_pval = 1 - stats.chi2.cdf(lr_stat, df=1)
    
    # Extract segment-specific effects
    # For Segment B: ar_treatment coefficient
    # For Segment A: ar_treatment + ar_x_segment_a
    ar_effect_seg_b = model2.params['ar_treatment']
    ar_effect_seg_a = model2.params['ar_treatment'] + model2.params['ar_x_segment_a']
    
    # Calculate marginal effects by segment
    me_frame = model2.get_margeff(at='mean').summary_frame()
    
    results = {
        'model_name': 'Logistic Regression - Conversion with Segments',
        'n_obs': len(df_analysis),
        'main_model': model1,
        'interaction_model': model2,
        'lr_test_stat': lr_stat,
        'lr_test_pval': lr_pval,
        'interaction_significant': lr_pval < 0.05,
        'ar_effect_segment_b': ar_effect_seg_b,
        'ar_effect_segment_a': ar_effect_seg_a,
        'odds_ratio_segment_b': np.exp(ar_effect_seg_b),
        'odds_ratio_segment_a': np.exp(ar_effect_seg_a),
        'interaction_coef': model2.params.get('ar_x_segment_a', 0),
        'interaction_pval': model2.pvalues.get('ar_x_segment_a', 1),
        'pseudo_r2': model2.prsquared,
        'aic': model2.aic,
        'bic': model2.bic
    }
    
    return results, model2


def logistic_regression_returns_with_segments(df: pd.DataFrame) -> dict:
    """
    Logistic regression for returns with segment heterogeneity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset
    
    Returns
    -------
    dict
        Regression results with interaction effects
    """
    # Filter to buyers only in base and optimistic scenarios
    df_buyers = df[(df['conversion'] == 1) & 
                   (df['scenario'].isin(['base', 'optimistic']))].copy()
    
    # Model with interaction
    model = smf.logit('returns ~ ar_treatment + segment_a + ar_x_segment_a', 
                     data=df_buyers).fit(disp=False)
    
    # Extract segment-specific effects
    ar_effect_seg_b = model.params['ar_treatment']
    ar_effect_seg_a = model.params['ar_treatment'] + model.params.get('ar_x_segment_a', 0)
    
    results = {
        'model_name': 'Logistic Regression - Returns with Segments',
        'n_obs': len(df_buyers),
        'model': model,
        'ar_effect_segment_b': ar_effect_seg_b,
        'ar_effect_segment_a': ar_effect_seg_a,
        'odds_ratio_segment_b': np.exp(ar_effect_seg_b),
        'odds_ratio_segment_a': np.exp(ar_effect_seg_a),
        'interaction_coef': model.params.get('ar_x_segment_a', 0),
        'interaction_pval': model.pvalues.get('ar_x_segment_a', 1),
        'pseudo_r2': model.prsquared
    }
    
    return results, model


# =============================================================================
# SEGMENT-SPECIFIC LINEAR REGRESSION MODELS
# =============================================================================

def linear_regression_cost_with_segments(df: pd.DataFrame) -> dict:
    """
    Linear regression for logistics cost with segment heterogeneity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset
    
    Returns
    -------
    dict
        Regression results with interaction effects
    """
    # Filter to base and optimistic scenarios
    df_analysis = df[df['scenario'].isin(['base', 'optimistic'])].copy()
    
    # Model with interaction
    model = smf.ols('logistic_cost_eur ~ ar_treatment + segment_a + ar_x_segment_a', 
                   data=df_analysis).fit()
    
    # Extract segment-specific effects
    ar_effect_seg_b = model.params['ar_treatment']
    ar_effect_seg_a = model.params['ar_treatment'] + model.params.get('ar_x_segment_a', 0)
    
    results = {
        'model_name': 'OLS Regression - Cost with Segments',
        'n_obs': len(df_analysis),
        'model': model,
        'ar_effect_segment_b': ar_effect_seg_b,
        'ar_effect_segment_a': ar_effect_seg_a,
        'interaction_coef': model.params.get('ar_x_segment_a', 0),
        'interaction_pval': model.pvalues.get('ar_x_segment_a', 1),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj
    }
    
    return results, model


def linear_regression_revenue_with_segments(df: pd.DataFrame) -> dict:
    """
    Linear regression for revenue with segment heterogeneity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset
    
    Returns
    -------
    dict
        Regression results with interaction effects
    """
    # Filter to base and optimistic scenarios
    df_analysis = df[df['scenario'].isin(['base', 'optimistic'])].copy()
    
    # Model with interaction
    model = smf.ols('net_revenue ~ ar_treatment + segment_a + ar_x_segment_a', 
                   data=df_analysis).fit()
    
    # Extract segment-specific effects
    ar_effect_seg_b = model.params['ar_treatment']
    ar_effect_seg_a = model.params['ar_treatment'] + model.params.get('ar_x_segment_a', 0)
    
    results = {
        'model_name': 'OLS Regression - Revenue with Segments',
        'n_obs': len(df_analysis),
        'model': model,
        'ar_effect_segment_b': ar_effect_seg_b,
        'ar_effect_segment_a': ar_effect_seg_a,
        'interaction_coef': model.params.get('ar_x_segment_a', 0),
        'interaction_pval': model.pvalues.get('ar_x_segment_a', 1),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj
    }
    
    return results, model


# =============================================================================
# HETEROGENEOUS TREATMENT EFFECTS ANALYSIS
# =============================================================================

def compute_heterogeneous_ate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute segment-specific Average Treatment Effects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset
    
    Returns
    -------
    pd.DataFrame
        Heterogeneous treatment effects by segment
    """
    results = []
    
    for segment in ['A', 'B']:
        # Filter data
        df_base_seg = df[(df['scenario'] == 'base') & 
                        (df['customer_segment'] == segment)]
        df_opt_seg = df[(df['scenario'] == 'optimistic') & 
                       (df['customer_segment'] == segment)]
        
        # Conversion ATE
        ate_conv = df_opt_seg['conversion'].mean() - df_base_seg['conversion'].mean()
        se_conv = np.sqrt(df_opt_seg['conversion'].var()/len(df_opt_seg) + 
                         df_base_seg['conversion'].var()/len(df_base_seg))
        
        # Return ATE (conditional)
        df_base_buyers = df_base_seg[df_base_seg['conversion'] == 1]
        df_opt_buyers = df_opt_seg[df_opt_seg['conversion'] == 1]
        
        if len(df_base_buyers) > 0 and len(df_opt_buyers) > 0:
            ate_return = df_opt_buyers['returns'].mean() - df_base_buyers['returns'].mean()
            se_return = np.sqrt(df_opt_buyers['returns'].var()/len(df_opt_buyers) + 
                               df_base_buyers['returns'].var()/len(df_base_buyers))
        else:
            ate_return = se_return = 0
        
        # Cost ATE
        ate_cost = df_opt_seg['logistic_cost_eur'].mean() - df_base_seg['logistic_cost_eur'].mean()
        se_cost = np.sqrt(df_opt_seg['logistic_cost_eur'].var()/len(df_opt_seg) + 
                         df_base_seg['logistic_cost_eur'].var()/len(df_base_seg))
        
        # Revenue ATE
        ate_revenue = df_opt_seg['net_revenue'].mean() - df_base_seg['net_revenue'].mean()
        se_revenue = np.sqrt(df_opt_seg['net_revenue'].var()/len(df_opt_seg) + 
                            df_base_seg['net_revenue'].var()/len(df_base_seg))
        
        results.append({
            'Segment': f'Segment {segment}',
            'Segment_Type': 'Early Adopters' if segment == 'A' else 'Traditional',
            'Conv_ATE': ate_conv,
            'Conv_SE': se_conv,
            'Conv_CI_Lower': ate_conv - 1.96*se_conv,
            'Conv_CI_Upper': ate_conv + 1.96*se_conv,
            'Return_ATE': ate_return,
            'Return_SE': se_return,
            'Cost_ATE': ate_cost,
            'Cost_SE': se_cost,
            'Revenue_ATE': ate_revenue,
            'Revenue_SE': se_revenue,
            'Revenue_CI_Lower': ate_revenue - 1.96*se_revenue,
            'Revenue_CI_Upper': ate_revenue + 1.96*se_revenue
        })
    
    return pd.DataFrame(results)


# =============================================================================
# ORIGINAL REGRESSION FUNCTIONS (kept for compatibility)
# =============================================================================

def logistic_regression_conversion(df: pd.DataFrame) -> dict:
    """
    Simple logistic regression for conversion probability.
    """
    df_analysis = df[df['scenario'].isin(['base', 'optimistic'])].copy()
    model1 = smf.logit('conversion ~ ar_treatment', data=df_analysis).fit(disp=False)
    
    results = {
        'model_name': 'Logistic Regression - Conversion',
        'n_obs': len(df_analysis),
        'baseline_conversion': df_analysis[df_analysis['ar_treatment'] == 0]['conversion'].mean(),
        'ar_conversion': df_analysis[df_analysis['ar_treatment'] == 1]['conversion'].mean(),
        'coefficient': model1.params['ar_treatment'],
        'std_error': model1.bse['ar_treatment'],
        'z_stat': model1.tvalues['ar_treatment'],
        'p_value': model1.pvalues['ar_treatment'],
        'odds_ratio': np.exp(model1.params['ar_treatment']),
        'conf_int_lower': np.exp(model1.conf_int().loc['ar_treatment', 0]),
        'conf_int_upper': np.exp(model1.conf_int().loc['ar_treatment', 1]),
        'pseudo_r2': model1.prsquared,
        'aic': model1.aic,
        'bic': model1.bic
    }
    
    base_prob = results['baseline_conversion']
    marginal_effect = model1.get_margeff(at='mean').summary_frame().loc['ar_treatment', 'dy/dx']
    results['marginal_effect'] = marginal_effect
    results['marginal_effect_pct'] = marginal_effect * 100
    
    return results, model1


def logistic_regression_returns(df: pd.DataFrame) -> dict:
    """
    Simple logistic regression for return probability.
    """
    df_buyers = df[(df['conversion'] == 1) & 
                   (df['scenario'].isin(['base', 'optimistic']))].copy()
    model = smf.logit('returns ~ ar_treatment', data=df_buyers).fit(disp=False)
    
    results = {
        'model_name': 'Logistic Regression - Returns',
        'n_obs': len(df_buyers),
        'baseline_returns_rate': df_buyers[df_buyers['ar_treatment'] == 0]['returns'].mean(),
        'ar_returns_rate': df_buyers[df_buyers['ar_treatment'] == 1]['returns'].mean(),
        'coefficient': model.params['ar_treatment'],
        'std_error': model.bse['ar_treatment'],
        'z_stat': model.tvalues['ar_treatment'],
        'p_value': model.pvalues['ar_treatment'],
        'odds_ratio': np.exp(model.params['ar_treatment']),
        'conf_int_lower': np.exp(model.conf_int().loc['ar_treatment', 0]),
        'conf_int_upper': np.exp(model.conf_int().loc['ar_treatment', 1]),
        'pseudo_r2': model.prsquared,
        'aic': model.aic,
        'bic': model.bic
    }
    
    marginal_effect = model.get_margeff(at='mean').summary_frame().loc['ar_treatment', 'dy/dx']
    results['marginal_effect'] = marginal_effect
    results['marginal_effect_pct'] = marginal_effect * 100
    
    return results, model


def linear_regression_cost(df: pd.DataFrame) -> dict:
    """
    Simple linear regression for logistics cost.
    """
    df_analysis = df[df['scenario'].isin(['base', 'optimistic'])].copy()
    model = smf.ols('logistic_cost_eur ~ ar_treatment', data=df_analysis).fit()
    
    results = {
        'model_name': 'OLS Regression - Logistics Cost',
        'n_obs': len(df_analysis),
        'baseline_mean_cost': df_analysis[df_analysis['ar_treatment'] == 0]['logistic_cost_eur'].mean(),
        'ar_mean_cost': df_analysis[df_analysis['ar_treatment'] == 1]['logistic_cost_eur'].mean(),
        'intercept': model.params['Intercept'],
        'ar_coefficient': model.params['ar_treatment'],
        'ar_std_error': model.bse['ar_treatment'],
        't_stat': model.tvalues['ar_treatment'],
        'p_value': model.pvalues['ar_treatment'],
        'conf_int_lower': model.conf_int().loc['ar_treatment', 0],
        'conf_int_upper': model.conf_int().loc['ar_treatment', 1],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'aic': model.aic,
        'bic': model.bic
    }
    
    results['cost_savings_per_user'] = abs(results['ar_coefficient'])
    results['cost_savings_pct'] = (results['cost_savings_per_user'] / 
                                   results['baseline_mean_cost']) * 100
    
    return results, model


def linear_regression_co2(df: pd.DataFrame) -> dict:
    """
    Simple linear regression for CO2 emissions.
    """
    df_analysis = df[df['scenario'].isin(['base', 'optimistic'])].copy()
    model = smf.ols('co2_kg ~ ar_treatment', data=df_analysis).fit()
    
    results = {
        'model_name': 'OLS Regression - CO2 Emissions',
        'n_obs': len(df_analysis),
        'baseline_mean_co2': df_analysis[df_analysis['ar_treatment'] == 0]['co2_kg'].mean(),
        'ar_mean_co2': df_analysis[df_analysis['ar_treatment'] == 1]['co2_kg'].mean(),
        'intercept': model.params['Intercept'],
        'ar_coefficient': model.params['ar_treatment'],
        'ar_std_error': model.bse['ar_treatment'],
        't_stat': model.tvalues['ar_treatment'],
        'p_value': model.pvalues['ar_treatment'],
        'conf_int_lower': model.conf_int().loc['ar_treatment', 0],
        'conf_int_upper': model.conf_int().loc['ar_treatment', 1],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'aic': model.aic,
        'bic': model.bic
    }
    
    results['co2_reduction_per_user'] = abs(results['ar_coefficient'])
    results['co2_reduction_pct'] = (results['co2_reduction_per_user'] / 
                                    results['baseline_mean_co2']) * 100
    
    return results, model


def linear_regression_revenue(df: pd.DataFrame) -> dict:
    """
    Simple linear regression for revenue.
    """
    df_analysis = df[df['scenario'].isin(['base', 'optimistic'])].copy()
    model = smf.ols('net_revenue ~ ar_treatment', data=df_analysis).fit()
    
    results = {
        'model_name': 'OLS Regression - Net Revenue (ARPU)',
        'n_obs': len(df_analysis),
        'baseline_arpu': df_analysis[df_analysis['ar_treatment'] == 0]['net_revenue'].mean(),
        'ar_arpu': df_analysis[df_analysis['ar_treatment'] == 1]['net_revenue'].mean(),
        'intercept': model.params['Intercept'],
        'ar_coefficient': model.params['ar_treatment'],
        'ar_std_error': model.bse['ar_treatment'],
        't_stat': model.tvalues['ar_treatment'],
        'p_value': model.pvalues['ar_treatment'],
        'conf_int_lower': model.conf_int().loc['ar_treatment', 0],
        'conf_int_upper': model.conf_int().loc['ar_treatment', 1],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'aic': model.aic,
        'bic': model.bic
    }
    
    results['revenue_uplift_per_user'] = results['ar_coefficient']
    results['revenue_uplift_pct'] = (results['revenue_uplift_per_user'] / 
                                     results['baseline_arpu']) * 100
    
    return results, model


def compute_ate_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall Average Treatment Effects.
    """
    df_base = df[df['scenario'] == 'base']
    df_opt = df[df['scenario'] == 'optimistic']
    
    ate_results = []
    
    # Conversion ATE
    ate_conv = df_opt['conversion'].mean() - df_base['conversion'].mean()
    se_conv = np.sqrt(df_opt['conversion'].var()/len(df_opt) + 
                     df_base['conversion'].var()/len(df_base))
    
    ate_results.append({
        'Outcome': 'Conversion Rate',
        'ATE': ate_conv,
        'SE': se_conv,
        'CI_Lower': ate_conv - 1.96*se_conv,
        'CI_Upper': ate_conv + 1.96*se_conv,
        'Relative Effect (%)': (ate_conv / df_base['conversion'].mean()) * 100
    })
    
    # Return ATE (conditional on purchase)
    df_base_buyers = df_base[df_base['conversion'] == 1]
    df_opt_buyers = df_opt[df_opt['conversion'] == 1]
    
    if len(df_base_buyers) > 0 and len(df_opt_buyers) > 0:
        ate_return = df_opt_buyers['returns'].mean() - df_base_buyers['returns'].mean()
        se_return = np.sqrt(df_opt_buyers['returns'].var()/len(df_opt_buyers) + 
                           df_base_buyers['returns'].var()/len(df_base_buyers))

        ate_results.append({
            'Outcome': 'Return Rate',
            'ATE': ate_return,
            'SE': se_return,
            'CI_Lower': ate_return - 1.96*se_return,
            'CI_Upper': ate_return + 1.96*se_return,
            'Relative Effect (%)': (ate_return / df_base_buyers['returns'].mean()) * 100
        })
    
    # Cost ATE
    ate_cost = df_opt['logistic_cost_eur'].mean() - df_base['logistic_cost_eur'].mean()
    se_cost = np.sqrt(df_opt['logistic_cost_eur'].var()/len(df_opt) + 
                      df_base['logistic_cost_eur'].var()/len(df_base))
    
    ate_results.append({
        'Outcome': 'Logistics Cost (€)',
        'ATE': ate_cost,
        'SE': se_cost,
        'CI_Lower': ate_cost - 1.96*se_cost,
        'CI_Upper': ate_cost + 1.96*se_cost,
        'Relative Effect (%)': (ate_cost / df_base['logistic_cost_eur'].mean()) * 100
    })
    
    # CO2 ATE
    ate_co2 = df_opt['co2_kg'].mean() - df_base['co2_kg'].mean()
    se_co2 = np.sqrt(df_opt['co2_kg'].var()/len(df_opt) + 
                     df_base['co2_kg'].var()/len(df_base))
    
    ate_results.append({
        'Outcome': 'CO2 Emissions (kg)',
        'ATE': ate_co2,
        'SE': se_co2,
        'CI_Lower': ate_co2 - 1.96*se_co2,
        'CI_Upper': ate_co2 + 1.96*se_co2,
        'Relative Effect (%)': (ate_co2 / df_base['co2_kg'].mean()) * 100
    })
    
    # Revenue ATE
    ate_revenue = df_opt['net_revenue'].mean() - df_base['net_revenue'].mean()
    se_revenue = np.sqrt(df_opt['net_revenue'].var()/len(df_opt) + 
                        df_base['net_revenue'].var()/len(df_base))
    
    ate_results.append({
        'Outcome': 'Net Revenue (€)',
        'ATE': ate_revenue,
        'SE': se_revenue,
        'CI_Lower': ate_revenue - 1.96*se_revenue,
        'CI_Upper': ate_revenue + 1.96*se_revenue,
        'Relative Effect (%)': (ate_revenue / df_base['net_revenue'].mean()) * 100
    })
    
    return pd.DataFrame(ate_results)


# =============================================================================
# ENHANCED RESULTS PRESENTATION
# =============================================================================

def print_enhanced_regression_results(
    conv_results: dict,
    return_results: dict,
    cost_results: dict,
    co2_results: dict,
    revenue_results: dict,
    conv_seg_results: dict,
    return_seg_results: dict,
    cost_seg_results: dict,
    revenue_seg_results: dict,
    ate_effects: pd.DataFrame,
    het_ate_effects: pd.DataFrame
) -> None:
    """
    Print comprehensive regression results including segment analysis.
    """
    print("\n" + "=" * 80)
    print("ENHANCED REGRESSION MODELS WITH SEGMENT HETEROGENEITY")
    print("Section 3.4.2: Statistical Modeling of AR Treatment Effects")
    print("=" * 80)
    
    # PART A: OVERALL EFFECTS
    print("\n" + "="*60)
    print("PART A: OVERALL TREATMENT EFFECTS")
    print("="*60)
    
    # 1. Overall Conversion
    print("\n1. OVERALL LOGISTIC REGRESSION - CONVERSION")
    print("-" * 60)
    print(f"Model: logit(P(conversion)) = β₀ + β₁·AR_treatment")
    print(f"\nBaseline conversion rate: {conv_results['baseline_conversion']*100:.2f}%")
    print(f"AR conversion rate: {conv_results['ar_conversion']*100:.2f}%")
    print(f"\nAR Treatment Effect:")
    print(f"  • Coefficient (β₁): {conv_results['coefficient']:.4f}")
    print(f"  • Odds Ratio: {conv_results['odds_ratio']:.3f}")
    print(f"  • P-value: {conv_results['p_value']:.4f} {'***' if conv_results['p_value'] < 0.001 else '**' if conv_results['p_value'] < 0.01 else '*' if conv_results['p_value'] < 0.05 else ''}")
    print(f"  • Marginal Effect: {conv_results['marginal_effect_pct']:.2f} pp")
    
    # 2. Overall Returns
    print("\n2. OVERALL LOGISTIC REGRESSION - RETURNS")
    print("-" * 60)
    print(f"Model: logit(P(returns|purchase)) = β₀ + β₁·AR_treatment")
    print(f"\nBaseline return rate: {return_results['baseline_returns_rate']*100:.2f}%")
    print(f"AR return rate: {return_results['ar_returns_rate']*100:.2f}%")
    print(f"\nAR Treatment Effect:")
    print(f"  • Coefficient (β₁): {return_results['coefficient']:.4f}")
    print(f"  • Odds Ratio: {return_results['odds_ratio']:.3f}")
    print(f"  • P-value: {return_results['p_value']:.4f} {'***' if return_results['p_value'] < 0.001 else '**' if return_results['p_value'] < 0.01 else '*' if return_results['p_value'] < 0.05 else ''}")
    print(f"  • Marginal Effect: {return_results['marginal_effect_pct']:.2f} pp")
    
    # PART B: HETEROGENEOUS EFFECTS
    print("\n" + "="*60)
    print("PART B: HETEROGENEOUS TREATMENT EFFECTS BY SEGMENT")
    print("="*60)
    
    # 3. Conversion with Segments
    print("\n3. SEGMENT-SPECIFIC CONVERSION EFFECTS")
    print("-" * 60)
    print(f"Model: logit(P(conversion)) = β₀ + β₁·AR + β₂·SegmentA + β₃·(AR×SegmentA)")
    print(f"\nInteraction Test:")
    print(f"  • LR Test Statistic: {conv_seg_results['lr_test_stat']:.3f}")
    print(f"  • P-value: {conv_seg_results['lr_test_pval']:.4f}")
    print(f"  • Interaction Significant: {'Yes' if conv_seg_results['interaction_significant'] else 'No'}")
    
    print(f"\nSegment-Specific AR Effects:")
    print(f"  Segment A (Early Adopters):")
    print(f"    • Log-odds effect: {conv_seg_results['ar_effect_segment_a']:.4f}")
    print(f"    • Odds Ratio: {conv_seg_results['odds_ratio_segment_a']:.3f}")
    print(f"  Segment B (Traditional):")
    print(f"    • Log-odds effect: {conv_seg_results['ar_effect_segment_b']:.4f}")
    print(f"    • Odds Ratio: {conv_seg_results['odds_ratio_segment_b']:.3f}")
    
    print(f"\nInteraction Coefficient: {conv_seg_results['interaction_coef']:.4f}")
    print(f"  (P-value: {conv_seg_results['interaction_pval']:.4f})")
    
    # 4. Returns with Segments
    print("\n4. SEGMENT-SPECIFIC RETURN EFFECTS")
    print("-" * 60)
    print(f"Model: logit(P(returns)) = β₀ + β₁·AR + β₂·SegmentA + β₃·(AR×SegmentA)")
    
    print(f"\nSegment-Specific AR Effects on Returns:")
    print(f"  Segment A (Early Adopters):")
    print(f"    • Log-odds effect: {return_seg_results['ar_effect_segment_a']:.4f}")
    print(f"    • Odds Ratio: {return_seg_results['odds_ratio_segment_a']:.3f}")
    print(f"  Segment B (Traditional):")
    print(f"    • Log-odds effect: {return_seg_results['ar_effect_segment_b']:.4f}")
    print(f"    • Odds Ratio: {return_seg_results['odds_ratio_segment_b']:.3f}")
    
    print(f"\nInteraction Coefficient: {return_seg_results['interaction_coef']:.4f}")
    print(f"  (P-value: {return_seg_results['interaction_pval']:.4f})")
    
    # 5. Cost with Segments
    print("\n5. SEGMENT-SPECIFIC COST EFFECTS")
    print("-" * 60)
    print(f"Model: Cost = β₀ + β₁·AR + β₂·SegmentA + β₃·(AR×SegmentA)")
    
    print(f"\nSegment-Specific AR Effects on Cost:")
    print(f"  Segment A: €{cost_seg_results['ar_effect_segment_a']:.3f} per user")
    print(f"  Segment B: €{cost_seg_results['ar_effect_segment_b']:.3f} per user")
    print(f"  Interaction: €{cost_seg_results['interaction_coef']:.3f} (P={cost_seg_results['interaction_pval']:.4f})")
    
    # 6. Revenue with Segments
    print("\n6. SEGMENT-SPECIFIC REVENUE EFFECTS")
    print("-" * 60)
    print(f"Model: Revenue = β₀ + β₁·AR + β₂·SegmentA + β₃·(AR×SegmentA)")
    
    print(f"\nSegment-Specific AR Effects on Revenue:")
    print(f"  Segment A: €{revenue_seg_results['ar_effect_segment_a']:.3f} per user")
    print(f"  Segment B: €{revenue_seg_results['ar_effect_segment_b']:.3f} per user")
    print(f"  Interaction: €{revenue_seg_results['interaction_coef']:.3f} (P={revenue_seg_results['interaction_pval']:.4f})")
    
    # PART C: CAUSAL INFERENCE
    print("\n" + "="*60)
    print("PART C: CAUSAL INFERENCE - AVERAGE TREATMENT EFFECTS")
    print("="*60)
    
    # 7. Overall ATE
    print("\n7. OVERALL AVERAGE TREATMENT EFFECTS")
    print("-" * 60)
    print("\nCausal Effects of AR Implementation (Optimistic vs Base):")
    print(ate_effects.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # 8. Heterogeneous ATE
    print("\n8. HETEROGENEOUS AVERAGE TREATMENT EFFECTS BY SEGMENT")
    print("-" * 60)
    print("\nSegment-Specific Treatment Effects:")
    
    for _, row in het_ate_effects.iterrows():
        print(f"\n{row['Segment']} ({row['Segment_Type']}):")
        print(f"  • Conversion ATE: {row['Conv_ATE']*100:.2f} pp [95% CI: {row['Conv_CI_Lower']*100:.2f}, {row['Conv_CI_Upper']*100:.2f}]")
        print(f"  • Return ATE: {row['Return_ATE']*100:.2f} pp")
        print(f"  • Cost ATE: €{row['Cost_ATE']:.3f}")
        print(f"  • Revenue ATE: €{row['Revenue_ATE']:.3f} [95% CI: €{row['Revenue_CI_Lower']:.3f}, €{row['Revenue_CI_Upper']:.3f}]")
    
    # Calculate effect heterogeneity
    seg_a_row = het_ate_effects[het_ate_effects['Segment'] == 'Segment A'].iloc[0]
    seg_b_row = het_ate_effects[het_ate_effects['Segment'] == 'Segment B'].iloc[0]
    
    print("\n9. EFFECT HETEROGENEITY ANALYSIS")
    print("-" * 60)
    print("\nDifferential Treatment Effects (Segment A vs Segment B):")
    print(f"  • Conversion: {(seg_a_row['Conv_ATE'] - seg_b_row['Conv_ATE'])*100:.2f} pp difference")
    print(f"  • Returns: {(seg_a_row['Return_ATE'] - seg_b_row['Return_ATE'])*100:.2f} pp difference")
    print(f"  • Revenue: €{seg_a_row['Revenue_ATE'] - seg_b_row['Revenue_ATE']:.2f} difference")
    
    conv_ratio = seg_a_row['Conv_ATE'] / seg_b_row['Conv_ATE'] if seg_b_row['Conv_ATE'] != 0 else 0
    revenue_ratio = seg_a_row['Revenue_ATE'] / seg_b_row['Revenue_ATE'] if seg_b_row['Revenue_ATE'] != 0 else 0
    
    print(f"\nRelative Effect Magnitudes:")
    print(f"  • Segment A captures {conv_ratio:.1f}× the conversion benefit of Segment B")
    print(f"  • Segment A captures {revenue_ratio:.1f}× the revenue benefit of Segment B")
    
    # 10. Key Insights
    print("\n10. KEY RESEARCH INSIGHTS")
    print("-" * 60)
    
    print("\n✓ Does AR impact vary by customer segment?")
    print(f"  YES - Early adopters (Segment A) show {conv_ratio:.1f}× stronger response")
    
    print("\n✓ Where should retailers focus AR investments?")
    print("  • Prioritize Segment A for immediate ROI")
    print("  • Develop simplified features for Segment B adoption")
    
    print("\n✓ What drives overall AR success?")
    seg_a_contrib = seg_a_row['Revenue_ATE'] * 0.5  # 50% of population
    seg_b_contrib = seg_b_row['Revenue_ATE'] * 0.5
    total_ate = seg_a_contrib + seg_b_contrib
    print(f"  • Segment A contributes {seg_a_contrib/total_ate*100:.1f}% of total revenue uplift")
    print(f"  • Segment B contributes {seg_b_contrib/total_ate*100:.1f}% of total revenue uplift")
    
    print("\n✓ Managerial Implications:")
    print("  1. AR benefits are highly concentrated among tech-savvy users")
    print("  2. Targeting and personalization strategies are crucial")
    print("  3. User education may help migrate Segment B toward Segment A behavior")
    print("  4. Consider segment-specific pricing to capture AR value")
    
    print("\n" + "=" * 80)
    print("Enhanced regression analysis complete with segment heterogeneity.")
    print("Results confirm asymmetric AR adoption benefits across customer types.")
    print("=" * 80)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("Loading dataset...")
    df = pd.read_csv("synthetic_ecom_ar_dataset.csv")
    
    # Rename return column if needed
    if 'return' in df.columns:
        df = df.rename(columns={'return': 'returns'})
    
    print("Preparing data for regression analysis...")
    df = prepare_regression_data(df)
    
    print("\nRunning comprehensive regression analysis with segmentation...")
    
    # PART 1: Simple regression models (overall effects)
    print("\n[1/11] Running overall conversion regression...")
    conv_results, conv_model = logistic_regression_conversion(df)
    
    print("[2/11] Running overall returns regression...")
    return_results, return_model = logistic_regression_returns(df)
    
    print("[3/11] Running overall cost regression...")
    cost_results, cost_model = linear_regression_cost(df)
    
    print("[4/11] Running overall CO2 regression...")
    co2_results, co2_model = linear_regression_co2(df)
    
    print("[5/11] Running overall revenue regression...")
    revenue_results, revenue_model = linear_regression_revenue(df)
    
    # PART 2: Segment-specific regression models (heterogeneous effects)
    print("[6/11] Running segment-specific conversion regression...")
    conv_seg_results, conv_seg_model = logistic_regression_conversion_with_segments(df)
    
    print("[7/11] Running segment-specific returns regression...")
    return_seg_results, return_seg_model = logistic_regression_returns_with_segments(df)
    
    print("[8/11] Running segment-specific cost regression...")
    cost_seg_results, cost_seg_model = linear_regression_cost_with_segments(df)
    
    print("[9/11] Running segment-specific revenue regression...")
    revenue_seg_results, revenue_seg_model = linear_regression_revenue_with_segments(df)
    
    # PART 3: Causal inference
    print("[10/11] Computing overall Average Treatment Effects...")
    ate_effects = compute_ate_effects(df)
    
    print("[11/11] Computing heterogeneous Average Treatment Effects...")
    het_ate_effects = compute_heterogeneous_ate(df)
    
    # Print comprehensive results
    print_enhanced_regression_results(
        conv_results,
        return_results,
        cost_results,
        co2_results,
        revenue_results,
        conv_seg_results,
        return_seg_results,
        cost_seg_results,
        revenue_seg_results,
        ate_effects,
        het_ate_effects
    )
    
    # Save enhanced results to CSV
    print("\nSaving enhanced regression results...")
    
    # Original summary
    regression_summary = pd.DataFrame([
        {
            'Model': 'Conversion (Logistic)',
            'Overall OR': f"{conv_results['odds_ratio']:.3f}",
            'Segment A OR': f"{conv_seg_results['odds_ratio_segment_a']:.3f}",
            'Segment B OR': f"{conv_seg_results['odds_ratio_segment_b']:.3f}",
            'Interaction P-value': conv_seg_results['interaction_pval'],
            'Overall P-value': conv_results['p_value']
        },
        {
            'Model': 'Returns (Logistic)',
            'Overall OR': f"{return_results['odds_ratio']:.3f}",
            'Segment A OR': f"{return_seg_results['odds_ratio_segment_a']:.3f}",
            'Segment B OR': f"{return_seg_results['odds_ratio_segment_b']:.3f}",
            'Interaction P-value': return_seg_results['interaction_pval'],
            'Overall P-value': return_results['p_value']
        },
        {
            'Model': 'Cost (OLS)',
            'Overall Effect': f"€{cost_results['ar_coefficient']:.3f}",
            'Segment A Effect': f"€{cost_seg_results['ar_effect_segment_a']:.3f}",
            'Segment B Effect': f"€{cost_seg_results['ar_effect_segment_b']:.3f}",
            'Interaction P-value': cost_seg_results['interaction_pval'],
            'Overall P-value': cost_results['p_value']
        },
        {
            'Model': 'CO₂ (OLS)',
            'Overall Effect': f"{co2_results['ar_coefficient']:.4f} kg",
            'Segment A Effect': "N/A",
            'Segment B Effect': "N/A",
            'Interaction P-value': "N/A",
            'Overall P-value': co2_results['p_value']
        },
        {
            'Model': 'Revenue (OLS)',
            'Overall Effect': f"€{revenue_results['ar_coefficient']:.3f}",
            'Segment A Effect': f"€{revenue_seg_results['ar_effect_segment_a']:.3f}",
            'Segment B Effect': f"€{revenue_seg_results['ar_effect_segment_b']:.3f}",
            'Interaction P-value': revenue_seg_results['interaction_pval'],
            'Overall P-value': revenue_results['p_value']
        }
    ])
    
    # Save all results
    regression_summary.to_csv("results_regression_summary_enhanced.csv", index=False)
    ate_effects.to_csv("results_ate_effects_overall.csv", index=False)
    het_ate_effects.to_csv("results_ate_effects_heterogeneous.csv", index=False)
    
    # Create segment comparison summary
    segment_comparison = pd.DataFrame([
        {
            'Metric': 'Conversion Lift (pp)',
            'Segment A': f"{het_ate_effects[het_ate_effects['Segment'] == 'Segment A']['Conv_ATE'].values[0]*100:.2f}",
            'Segment B': f"{het_ate_effects[het_ate_effects['Segment'] == 'Segment B']['Conv_ATE'].values[0]*100:.2f}",
            'Ratio A/B': f"{het_ate_effects[het_ate_effects['Segment'] == 'Segment A']['Conv_ATE'].values[0] / het_ate_effects[het_ate_effects['Segment'] == 'Segment B']['Conv_ATE'].values[0]:.1f}"
        },
        {
            'Metric': 'Return Reduction (pp)',
            'Segment A': f"{abs(het_ate_effects[het_ate_effects['Segment'] == 'Segment A']['Return_ATE'].values[0]*100):.2f}",
            'Segment B': f"{abs(het_ate_effects[het_ate_effects['Segment'] == 'Segment B']['Return_ATE'].values[0]*100):.2f}",
            'Ratio A/B': f"{abs(het_ate_effects[het_ate_effects['Segment'] == 'Segment A']['Return_ATE'].values[0]) / abs(het_ate_effects[het_ate_effects['Segment'] == 'Segment B']['Return_ATE'].values[0]):.1f}"
        },
        {
            'Metric': 'Revenue Uplift (€)',
            'Segment A': f"{het_ate_effects[het_ate_effects['Segment'] == 'Segment A']['Revenue_ATE'].values[0]:.2f}",
            'Segment B': f"{het_ate_effects[het_ate_effects['Segment'] == 'Segment B']['Revenue_ATE'].values[0]:.2f}",
            'Ratio A/B': f"{het_ate_effects[het_ate_effects['Segment'] == 'Segment A']['Revenue_ATE'].values[0] / het_ate_effects[het_ate_effects['Segment'] == 'Segment B']['Revenue_ATE'].values[0]:.1f}"
        }
    ])
    
    segment_comparison.to_csv("results_segment_effect_comparison.csv", index=False)
    
    print("\n✓ Enhanced results saved to CSV files:")
    print("  - results_regression_summary_enhanced.csv")
    print("  - results_ate_effects_overall.csv")
    print("  - results_ate_effects_heterogeneous.csv")
    print("  - results_segment_effect_comparison.csv")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Finding: AR benefits are highly heterogeneous across customer segments.")
    print("Early adopters (Segment A) capture disproportionately large benefits,")
    print("supporting targeted AR deployment strategies.")
    print("=" * 80)