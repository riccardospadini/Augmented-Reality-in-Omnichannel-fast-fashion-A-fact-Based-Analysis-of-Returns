"""
Chapter 4 Figures overview
Thesis: "Augmented Reality in Omnichannel Fast Fashion: A Fact-Based Impact of Return on Customer Satisfaction"
Author: Riccardo Spadini
Date: October 2025
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ---------- Setup ----------
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Enhanced style settings for thesis quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",  # More academic
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color scheme for consistency
COLORS = {
    'pessimistic': '#E74C3C',  # Red
    'base': '#95A5A6',          # Gray  
    'optimistic': '#27AE60',    # Green
    'segment_a': '#3498DB',     # Blue
    'segment_b': '#F39C12',     # Orange
    'kept': '#2C3E50',          # Dark blue
    'returned': '#C0392B',      # Dark red
}

def savefig_pdf(name: str, tight=True):
    """Save figure as PDF with proper layout"""
    if tight:
        plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{name}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUTDIR, f"{name}.png"), bbox_inches='tight', dpi=300)  # Also save PNG
    print(f"  ✓ Saved: {name}.pdf and {name}.png")
    plt.close()

# ---------- Data from your analysis ----------
scenarios = ["Pessimistic", "Base", "Optimistic"]
scenarios_short = ["Pess.", "Base", "Opt."]

# Overall KPIs
conversion_overall = [2.08, 2.96, 5.07]  # %
returns_overall    = [39.62, 32.85, 8.66]  # % on buyers
aov_mean           = [86.36, 102.40, 122.88]  # €
aov_sd             = [30.34, 39.99, 47.75]    # €
cost_per_order     = [17.96, 17.45, 8.62]     # €
co2_per_order      = [1.121, 1.098, 1.026]    # kg

# Segment-specific metrics
conv_A = [2.56, 3.45, 6.11]   # %
conv_B = [1.62, 2.48, 4.03]   # %
ret_A  = [39.86, 31.82, 5.01]  # %
ret_B  = [39.24, 34.27, 14.17] # %

# Cost asymmetry
kept_orders   = [1259, 1989, 4631]
kept_cost     = [5.00, 5.00, 5.00]
returned_orders = [826, 973, 439]
returned_cost   = [37.70, 42.89, 46.75]

# CO2 metrics
buyers = [2085, 2962, 5070]
co2_total = [2337, 3252, 5202]  # kg per 100k users
co2_per_user = [0.023, 0.033, 0.052]   # kg/user

# Sensitivity data
ashare = np.array([20,30,40,50,60,70,80])
ashare_conv_lift = np.array([1.45, 1.49, 1.59, 1.89, 2.01, 2.27, 2.15])
ashare_ret_red   = np.array([18.74,17.57,20.00,18.39,16.66,21.41,24.95])
ashare_rev_uplift= np.array([2.12,2.30,2.39,2.91,3.11,3.41,3.38])

# =============================================================================
# MUST-HAVE FIGURES (Priority 1-5)
# =============================================================================

def figure1_conversion_returns():
    """Figure 1: Conversion & Returns by Scenario - THE headline result"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Conversion bars (primary axis)
    bars1 = ax1.bar(x - width/2, conversion_overall, width, 
                    label='Conversion Rate', 
                    color=[COLORS['pessimistic'], COLORS['base'], COLORS['optimistic']],
                    edgecolor='black', linewidth=1.5)
    
    # Add value labels on conversion bars
    for bar, val in zip(bars1, conversion_overall):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 6)
    
    # Return bars (secondary axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, returns_overall, width,
                    label='Return Rate (on buyers)',
                    color=['#ffcccc', '#cccccc', '#ccffcc'],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on return bars
    for bar, val in zip(bars2, returns_overall):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Return Rate on Buyers (%)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 50)
    
    # Styling
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=11, fontweight='bold')
    ax1.set_title('AR Impact on Conversion and Return Rates\n(Key Performance Indicators)',
                  fontsize=14, fontweight='bold', pad=20)
    
    # Add annotations for AR effect
    ax1.annotate('', xy=(x[2]-0.1, 5.5), xytext=(x[1]+0.1, 3.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text(1.5, 4.5, '+71% conversion', fontsize=10, color='green', fontweight='bold')
    
    ax2.annotate('', xy=(x[2]+0.1, 10), xytext=(x[1]-0.1, 30),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.text(1.5, 20, '-74% returns', fontsize=10, color='green', fontweight='bold')
    
    # Legends
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    savefig_pdf("fig1_conversion_returns_headline")
    return fig

def figure2_conversion_by_segment():
    """Figure 2: Conversion by Scenario × Segment - Shows heterogeneity"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Segment A bars
    bars1 = ax.bar(x - width/2, conv_A, width, 
                   label='Segment A (Early Adopters)',
                   color=COLORS['segment_a'], edgecolor='black', linewidth=1.5)
    
    # Segment B bars
    bars2 = ax.bar(x + width/2, conv_B, width,
                   label='Segment B (Traditional)',
                   color=COLORS['segment_b'], edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.08,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight='bold')
    ax.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Heterogeneous AR Effects: Conversion by Customer Segment',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 7)
    
    # Add lift annotations
    for i, scenario in enumerate(scenarios):
        if i > 0:  # Compare to base
            lift_a = (conv_A[i] - conv_A[1]) / conv_A[1] * 100
            lift_b = (conv_B[i] - conv_B[1]) / conv_B[1] * 100
            if i == 2:  # Optimistic
                ax.text(i, 6.5, f'A: +{lift_a:.0f}%\nB: +{lift_b:.0f}%',
                       ha='center', fontsize=9, color='green', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    savefig_pdf("fig2_conversion_by_segment")
    return fig

def figure3_returns_by_segment():
    """Figure 3: Returns by Scenario × Segment - Shows differential reduction"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Segment A bars
    bars1 = ax.bar(x - width/2, ret_A, width,
                   label='Segment A (Early Adopters)',
                   color=COLORS['segment_a'], edgecolor='black', linewidth=1.5)
    
    # Segment B bars
    bars2 = ax.bar(x + width/2, ret_B, width,
                   label='Segment B (Traditional)',
                   color=COLORS['segment_b'], edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=10)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight='bold')
    ax.set_ylabel('Return Rate on Buyers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Heterogeneous AR Effects: Return Rates by Customer Segment',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 45)
    
    # Add reduction annotations for Optimistic scenario
    base_idx = 1
    opt_idx = 2
    reduction_a = ret_A[base_idx] - ret_A[opt_idx]
    reduction_b = ret_B[base_idx] - ret_B[opt_idx]
    
    ax.text(opt_idx, 25, f'Reduction from Base:\nA: -{reduction_a:.1f}pp (-{reduction_a/ret_A[base_idx]*100:.0f}%)\nB: -{reduction_b:.1f}pp (-{reduction_b/ret_B[base_idx]*100:.0f}%)',
            ha='center', fontsize=9, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    savefig_pdf("fig3_returns_by_segment")
    return fig

def figure4_cost_asymmetry():
    """Figure 4: Cost Asymmetry - Shows why return reduction matters financially"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Left panel: Average costs
    bars1 = ax1.bar(x - width/2, kept_cost, width,
                    label='Kept Orders',
                    color=COLORS['kept'], edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, returned_cost, width,
                    label='Returned Orders',
                    color=COLORS['returned'], edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'€{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=11)
    ax1.set_ylabel('Average Logistics Cost (€)', fontsize=12, fontweight='bold')
    ax1.set_title('Cost per Order Type', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 55)
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right panel: Cost multiplier visualization
    multipliers = [r/k for r, k in zip(returned_cost, kept_cost)]
    colors_mult = [COLORS['pessimistic'], COLORS['base'], COLORS['optimistic']]
    bars3 = ax2.bar(scenarios, multipliers, color=colors_mult, 
                    edgecolor='black', linewidth=1.5)
    
    for bar, mult in zip(bars3, multipliers):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mult:.1f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Cost Multiplier (Returned/Kept)', fontsize=12, fontweight='bold')
    ax2.set_title('Return Cost Premium', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 12)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add insight box
    ax2.text(0.5, 10, 'Key Insight:\nReturned orders cost\n7-9× more than kept orders\n→ Small return reduction\n= Large cost savings',
             transform=ax2.transData, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    fig.suptitle('Cost Asymmetry: Why Return Reduction Drives Profitability',
                 fontsize=14, fontweight='bold')
    
    savefig_pdf("fig4_cost_asymmetry")
    return fig

def figure5_co2_analysis():
    """Figure 5: CO2 Analysis - Intensity vs Totals paradox"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(scenarios))
    
    # Left panel: CO2 per order (intensity)
    colors_co2 = [COLORS['pessimistic'], COLORS['base'], COLORS['optimistic']]
    bars1 = ax1.bar(scenarios, co2_per_order, color=colors_co2,
                    edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars1, co2_per_order):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('CO₂ per Order (kg)', fontsize=12, fontweight='bold')
    ax1.set_title('Carbon Intensity\n(Efficiency Improvement)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.3)
    
    # Add trend arrow
    ax1.annotate('', xy=(2.2, 1.0), xytext=(0.8, 1.12),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text(1.5, 1.15, '-8.4%', fontsize=11, color='green', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right panel: Total CO2 and order volume
    ax2_twin = ax2.twinx()
    
    # Total CO2 bars
    bars2 = ax2.bar(x - 0.2, co2_total, 0.4,
                    label='Total CO₂ (kg)',
                    color='lightgray', edgecolor='black', linewidth=1.5)
    
    # Number of buyers line
    line = ax2_twin.plot(scenarios, buyers, 'o-', color='red', linewidth=2,
                        markersize=8, label='Number of Buyers')
    
    # Labels
    ax2.set_ylabel('Total CO₂ (kg per 100k users)', fontsize=12, fontweight='bold')
    ax2_twin.set_ylabel('Number of Buyers', fontsize=12, fontweight='bold', color='red')
    ax2.set_title('Volume Effect\n(More Orders Despite Lower Intensity)', 
                  fontsize=13, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('CO₂ Impact Analysis: The Efficiency-Volume Trade-off',
                 fontsize=14, fontweight='bold')
    
    savefig_pdf("fig5_co2_analysis")
    return fig

# =============================================================================
# HIGH-IMPACT STRATEGIC FIGURES (Priority 6-7)
# =============================================================================

def figure6_sensitivity_ashare():
    """Figure 6: Sensitivity to Segment A share - Strategic insight"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Revenue uplift (primary axis)
    line1 = ax1.plot(ashare, ashare_rev_uplift, 'o-', linewidth=2, 
                     color=COLORS['optimistic'], markersize=8,
                     label='Revenue Uplift')
    
    ax1.set_xlabel('Share of Segment A (Early Adopters) %', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue Uplift (€/user)', fontsize=12, fontweight='bold', 
                   color=COLORS['optimistic'])
    ax1.tick_params(axis='y', labelcolor=COLORS['optimistic'])
    
    # Fill area under revenue curve
    ax1.fill_between(ashare, 0, ashare_rev_uplift, alpha=0.2, color=COLORS['optimistic'])
    
    # Return reduction (secondary axis)
    ax2 = ax1.twinx()
    line2 = ax2.plot(ashare, ashare_ret_red, 's--', linewidth=2,
                     color=COLORS['segment_a'], markersize=7,
                     label='Return Reduction')
    
    ax2.set_ylabel('Return Reduction (pp)', fontsize=12, fontweight='bold',
                   color=COLORS['segment_a'])
    ax2.tick_params(axis='y', labelcolor=COLORS['segment_a'])
    
    # Title and styling
    ax1.set_title('Sensitivity Analysis: Impact of Customer Mix\n(More Early Adopters → Higher Value)',
                  fontsize=14, fontweight='bold', pad=20)
    
    # Add optimal zone
    ax1.axvspan(60, 80, alpha=0.1, color='green')
    ax1.text(70, 2.0, 'Optimal\nZone', ha='center', fontsize=10, 
             fontweight='bold', color='green')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(15, 85)
    
    savefig_pdf("fig6_sensitivity_ashare")
    return fig

def figure7_adoption_scenarios():
    """Figure 7: Differential Adoption Scenarios - Implementation strategy"""
    
    # Data
    adopt_scenarios = ["A=100%\nB=20%", "A=100%\nB=50%", "Full\n(100/100)",
                      "A=80%\nB=30%", "A=50%\nB=50%"]
    adopt_rev_uplift = [2.04, 2.40, 3.24, 1.80, 1.01]
    adopt_benefit = [50.94, 60.07, 80.95, 44.90, 25.30]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(adopt_scenarios))
    
    # Left panel: Revenue uplift
    colors = ['#2ecc71', '#27ae60', '#16a085', '#f39c12', '#e74c3c']
    bars1 = ax1.bar(x, adopt_rev_uplift, color=colors, 
                    edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars1, adopt_rev_uplift):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'€{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add reference line for full adoption
    full_idx = 2
    ax1.axhline(y=adopt_rev_uplift[full_idx], color='gray', 
                linestyle='--', alpha=0.5, label='Full Adoption')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(adopt_scenarios, fontsize=10)
    ax1.set_ylabel('Revenue Uplift (€/user)', fontsize=12, fontweight='bold')
    ax1.set_title('Revenue Impact by Adoption Scenario', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 3.8)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right panel: Benefit realization %
    bars2 = ax2.bar(x, adopt_benefit, color=colors,
                    edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars2, adopt_benefit):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(adopt_scenarios, fontsize=10)
    ax2.set_ylabel('Benefit Realization (%)', fontsize=12, fontweight='bold')
    ax2.set_title('% of Maximum Benefit Captured', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add insight annotation
    ax2.text(3, 60, 'Key Insight:\nRealistic scenario\n(A=80%, B=30%)\ncaptures ~45%\nof max benefit',
             fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    fig.suptitle('Differential AR Adoption: Strategic Rollout Scenarios',
                 fontsize=14, fontweight='bold')
    
    savefig_pdf("fig7_adoption_scenarios")
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def generate_priority_figures(selection='complete'):
    """
    Generate figures based on priority selection
    
    Parameters:
    -----------
    selection : str
        'compact' - 3 must-have figures only
        'complete' - 5-6 priority figures
        'full' - all 7 figures
    """
    
    print("=" * 60)
    print("GENERATING CHAPTER 4 FIGURES")
    print(f"Selection: {selection.upper()}")
    print("=" * 60)
    
    figures_generated = []
    
    if selection in ['compact', 'complete', 'full']:
        # Must-have figures (1-3 for compact, 1-5 for complete/full)
        print("\n[MUST-HAVE FIGURES]")
        
        print("\n1. Generating Conversion & Returns (Headline Result)...")
        figure1_conversion_returns()
        figures_generated.append("Figure 1: Conversion & Returns by Scenario")
        
        if selection in ['complete', 'full']:
            print("\n2. Generating Conversion by Segment...")
            figure2_conversion_by_segment()
            figures_generated.append("Figure 2: Conversion by Segment")
            
            print("\n3. Generating Returns by Segment...")
            figure3_returns_by_segment()
            figures_generated.append("Figure 3: Returns by Segment")
        
        print("\n4. Generating Cost Asymmetry...")
        figure4_cost_asymmetry()
        figures_generated.append("Figure 4: Cost Asymmetry")
        
        print("\n5. Generating CO₂ Analysis...")
        figure5_co2_analysis()
        figures_generated.append("Figure 5: CO₂ Impact")
    
    if selection == 'full':
        # Strategic figures
        print("\n[STRATEGIC FIGURES]")
        
        print("\n6. Generating Sensitivity to A-share...")
        figure6_sensitivity_ashare()
        figures_generated.append("Figure 6: Sensitivity to Segment Mix")
        
        print("\n7. Generating Adoption Scenarios...")
        figure7_adoption_scenarios()
        figures_generated.append("Figure 7: Adoption Scenarios")
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    
    print(f"\n✓ Generated {len(figures_generated)} figures:")
    for fig in figures_generated:
        print(f"  • {fig}")
    
    print(f"\n📁 Files saved in: {os.path.abspath(OUTDIR)}/")
    print("  Formats: PDF (vector) + PNG (raster)")
    
    # LaTeX template
    print("\n" + "-" * 60)
    print("LATEX INTEGRATION EXAMPLE:")
    print("-" * 60)
    print("""
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{figures/fig1_conversion_returns_headline.pdf}
    \\caption{Impact of AR implementation on conversion and return rates across 
    three scenarios. The optimistic scenario (with AR) shows a 71\\% increase in 
    conversion rate and a 74\\% reduction in returns compared to the base scenario.}
    \\label{fig:conversion_returns}
\\end{figure}
    """)
    
    return figures_generated


if __name__ == "__main__":
    # Choose your selection:
    # 'compact' = 3 figures (bare minimum)
    # 'complete' = 5 figures (recommended)
    # 'full' = 7 figures (comprehensive)
    
    selection = 'complete'  # Change this based on your needs
    
    figures = generate_priority_figures(selection=selection)
    
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 60)
    
    if selection == 'compact':
        print("Compact selection (3 figures) covers:")
        print("  • Main effect (conversion/returns)")
        print("  • Financial impact (cost asymmetry)")
        print("  • Sustainability angle (CO₂)")
        print("\nIdeal for: Executive summary or conference presentation")
        
    elif selection == 'complete':
        print("Complete selection (5 figures) covers:")
        print("  • Overall effects + segment heterogeneity")
        print("  • Economic and environmental impacts")
        print("\nIdeal for: Main thesis chapter or detailed presentation")
        
    else:  # full
        print("Full selection (7 figures) includes:")
        print("  • All core analyses + strategic sensitivity")
        print("\nIdeal for: Comprehensive documentation or appendix")
    
    print("\n📊 SUGGESTED PLACEMENT IN THESIS:")
    print("-" * 60)
    print("Section 4.1: Figure 1 (Headline results)")
    print("Section 4.2: Figures 2-3 (Segment analysis)")
    print("Section 4.3: Figure 4 (Cost analysis)")
    print("Section 4.4: Figure 5 (Environmental impact)")
    print("Section 4.5: Figures 6-7 (Sensitivity & Strategy)")
    
    print("\n✅ Done. Figures saved to:", OUTDIR)
    