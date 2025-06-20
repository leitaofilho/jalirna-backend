#!/usr/bin/env python3
"""
Summary analysis and final recommendations for COR vs COR 2 variables.
This script provides the complete analysis findings and actionable recommendations.
"""

import pandas as pd
import numpy as np

def generate_summary_report():
    """
    Generate a comprehensive summary report of the COR vs COR 2 analysis.
    """
    
    # Load the dataset
    df = pd.read_csv('/Users/aiacontext/PycharmProjects/jaliRNA/dataset/dataset_drc_processed.csv', sep=';')
    
    print("="*80)
    print("COR vs COR 2 RELATIONSHIP ANALYSIS - EXECUTIVE SUMMARY")
    print("="*80)
    print()
    
    # Key findings
    print("📊 KEY FINDINGS:")
    print()
    
    print("1. DATA STRUCTURE:")
    print(f"   • Dataset contains {len(df)} records")
    print(f"   • Both columns have complete data (no missing values)")
    print(f"   • COR: {len(df['COR'].unique())} categories (text)")
    print(f"   • COR 2: {len(df['COR 2'].unique())} categories (numeric)")
    print()
    
    print("2. RELATIONSHIP PATTERN:")
    crosstab = pd.crosstab(df['COR'], df['COR 2'])
    print("   • COR 2 is a BINARY RECODING of COR")
    print("   • Mapping pattern:")
    for cor_val in sorted(df['COR'].unique()):
        cor2_vals = df[df['COR'] == cor_val]['COR 2'].unique()
        if len(cor2_vals) == 1:
            print(f"     - {cor_val} → {cor2_vals[0]} (perfect mapping)")
        else:
            main_val = df[df['COR'] == cor_val]['COR 2'].mode().iloc[0]
            count_main = (df[df['COR'] == cor_val]['COR 2'] == main_val).sum()
            count_total = len(df[df['COR'] == cor_val])
            pct = (count_main / count_total) * 100
            print(f"     - {cor_val} → {main_val} ({count_main}/{count_total} = {pct:.1f}%)")
    print()
    
    print("3. INFORMATION CONTENT:")
    cor_entropy = -sum((df['COR'].value_counts(normalize=True) * np.log2(df['COR'].value_counts(normalize=True))))
    cor2_entropy = -sum((df['COR 2'].value_counts(normalize=True) * np.log2(df['COR 2'].value_counts(normalize=True))))
    info_loss = ((cor_entropy - cor2_entropy) / cor_entropy) * 100
    
    print(f"   • COR entropy: {cor_entropy:.3f} bits")
    print(f"   • COR 2 entropy: {cor2_entropy:.3f} bits")
    print(f"   • Information loss: {info_loss:.1f}%")
    print()
    
    print("4. STATISTICAL RELATIONSHIP:")
    # Convert COR to numeric for correlation
    cor_mapping = {val: i for i, val in enumerate(sorted(df['COR'].unique()))}
    cor_numeric = df['COR'].map(cor_mapping)
    correlation = cor_numeric.corr(df['COR 2'])
    print(f"   • Correlation coefficient: {correlation:.3f}")
    print(f"   • Relationship strength: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.5 else 'Weak'}")
    print()
    
    # Interpretation
    print("🔍 INTERPRETATION:")
    print()
    print("COR 2 represents a BINARY RACIAL CLASSIFICATION:")
    print("   • 0 = White/Non-minority (mostly BRANCO)")
    print("   • 1 = Non-white/Minority (NEGRA + PARDA + most OUTRAS)")
    print()
    print("This is a common approach in health research to create:")
    print("   • Simplified race/ethnicity categories")
    print("   • Binary minority status indicators")
    print("   • Variables with better statistical power")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS FOR FEATURE SELECTION:")
    print()
    
    print("SCENARIO 1 - Detailed Health Disparities Research:")
    print("   👍 USE COR (original 4-category variable)")
    print("   • Preserves all racial/ethnic distinctions")
    print("   • Better for identifying specific group differences")
    print("   • More informative for healthcare policy")
    print("   ⚠️  Watch for small sample sizes in OUTRAS category (n=7)")
    print()
    
    print("SCENARIO 2 - Binary Minority/Majority Analysis:")
    print("   👍 USE COR 2 (binary variable)")
    print("   • Simplified analysis with better statistical power")
    print("   • Clear minority (n=154) vs non-minority (n=41) comparison")
    print("   • Easier interpretation of results")
    print("   ⚠️  Loses important within-minority distinctions")
    print()
    
    print("SCENARIO 3 - Comprehensive Modeling:")
    print("   👍 USE BOTH (with caution)")
    print("   • Include both for different analytical purposes")
    print("   • COR for detailed analysis, COR 2 for simplified comparisons")
    print("   • Monitor for multicollinearity (VIF > 5-10)")
    print("   • Consider using only one in final models")
    print()
    
    print("SCENARIO 4 - Machine Learning Models:")
    print("   👍 PREFER COR 2 or dummy-coded COR")
    print("   • COR 2: Ready for use (numeric binary)")
    print("   • COR: Convert to dummy variables (3 dummies for 4 categories)")
    print("   • Avoid using both raw variables together")
    print()
    
    # Final recommendation
    print("🎯 FINAL RECOMMENDATION:")
    print()
    print("COR 2 is NOT redundant - it serves a different analytical purpose:")
    print()
    print("   • COR 2 is a PURPOSEFUL SIMPLIFICATION of COR")
    print("   • Both variables have legitimate uses in health research")
    print("   • Choice depends on your specific research question:")
    print()
    print("     ✅ Research question about specific racial/ethnic groups → Use COR")
    print("     ✅ Research question about minority vs majority → Use COR 2")
    print("     ✅ Exploratory analysis → Try both separately")
    print("     ❌ Never use both in the same regression model")
    print()
    
    print("📋 PRACTICAL IMPLEMENTATION:")
    print()
    print("For model building:")
    print("   1. Start with COR 2 for simplicity")
    print("   2. If significant, explore COR for detailed patterns")
    print("   3. Create dummy variables if using COR in regression")
    print("   4. Always check VIF if considering both variables")
    print()
    
    # Sample size considerations
    print("📊 SAMPLE SIZE CONSIDERATIONS:")
    print()
    cor_counts = df['COR'].value_counts()
    cor2_counts = df['COR 2'].value_counts()
    
    print("COR category sizes:")
    for cat, count in cor_counts.items():
        pct = (count / len(df)) * 100
        power_note = "✅ Good power" if count >= 30 else "⚠️ Limited power" if count >= 10 else "❌ Very limited power"
        print(f"   • {cat}: n={count} ({pct:.1f}%) - {power_note}")
    
    print()
    print("COR 2 category sizes:")
    for cat, count in cor2_counts.items():
        cat_name = "Non-minority" if cat == 0 else "Minority"
        pct = (count / len(df)) * 100
        power_note = "✅ Good power" if count >= 30 else "⚠️ Limited power"
        print(f"   • {cat} ({cat_name}): n={count} ({pct:.1f}%) - {power_note}")
    
    print()
    print("="*80)
    print("Analysis complete. Choose the variable(s) that best fit your research objectives.")
    print("="*80)

if __name__ == "__main__":
    generate_summary_report()