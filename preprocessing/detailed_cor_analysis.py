#!/usr/bin/env python3
"""
Detailed analysis of COR and COR 2 relationship with additional insights.
"""

import pandas as pd
import numpy as np

def detailed_cor_analysis():
    """
    Provide detailed analysis with clearer interpretation.
    """
    
    # Load the dataset
    df = pd.read_csv('/Users/aiacontext/PycharmProjects/jaliRNA/dataset/dataset_drc_processed.csv', sep=';')
    
    print("=== DETAILED COR vs COR 2 RELATIONSHIP ANALYSIS ===")
    print(f"Dataset size: {len(df)} records")
    print()
    
    # Detailed mapping analysis
    print("=== DETAILED MAPPING PATTERNS ===")
    
    crosstab = pd.crosstab(df['COR'], df['COR 2'], margins=True)
    crosstab_pct = pd.crosstab(df['COR'], df['COR 2'], normalize='index') * 100
    
    print("Absolute counts:")
    print(crosstab)
    print()
    
    print("Percentage within each COR category:")
    print(crosstab_pct.round(1))
    print()
    
    # Analyze the encoding pattern
    print("=== ENCODING PATTERN INTERPRETATION ===")
    
    # Examine what COR 2 values represent
    print("COR 2 = 0 contains:")
    cor2_0_breakdown = df[df['COR 2'] == 0]['COR'].value_counts()
    for cor_val, count in cor2_0_breakdown.items():
        pct = (count / len(df[df['COR 2'] == 0])) * 100
        print(f"  - {cor_val}: {count} records ({pct:.1f}%)")
    
    print("\nCOR 2 = 1 contains:")
    cor2_1_breakdown = df[df['COR 2'] == 1]['COR'].value_counts()
    for cor_val, count in cor2_1_breakdown.items():
        pct = (count / len(df[df['COR 2'] == 1])) * 100
        print(f"  - {cor_val}: {count} records ({pct:.1f}%)")
    print()
    
    # Analyze the binary encoding logic
    print("=== BINARY ENCODING LOGIC ANALYSIS ===")
    
    # It appears COR 2 might be a binary encoding
    # Let's see the pattern more clearly
    print("Apparent encoding pattern:")
    print("- COR 2 = 0: Mostly 'BRANCO' (White) + 1 'OUTRAS' (Other)")
    print("- COR 2 = 1: 'NEGRA' (Black) + 'PARDA' (Brown/Mixed) + most 'OUTRAS' (Other)")
    print()
    
    print("This suggests COR 2 might be a binary classification:")
    print("- 0: Non-minority/White-coded")
    print("- 1: Minority/Non-white-coded")
    print()
    
    # Information loss analysis
    print("=== INFORMATION LOSS ANALYSIS ===")
    
    original_categories = len(df['COR'].unique())
    binary_categories = len(df['COR 2'].unique())
    
    print(f"Original COR has {original_categories} categories: {sorted(df['COR'].unique())}")
    print(f"COR 2 has {binary_categories} categories: {sorted(df['COR 2'].unique())}")
    print(f"Information reduction: {original_categories} -> {binary_categories} categories")
    print()
    
    # Calculate information preservation
    print("=== INFORMATION PRESERVATION ANALYSIS ===")
    
    # For each original category, show how it's distributed in the binary version
    print("Information preservation by category:")
    for cor_val in sorted(df['COR'].unique()):
        subset = df[df['COR'] == cor_val]
        cor2_dist = subset['COR 2'].value_counts().sort_index()
        
        if len(cor2_dist) == 1:
            preservation = "PERFECT"
            main_mapping = cor2_dist.index[0]
        else:
            # Calculate how concentrated the mapping is
            max_pct = (cor2_dist.max() / cor2_dist.sum()) * 100
            if max_pct >= 80:
                preservation = f"GOOD ({max_pct:.1f}%)"
                main_mapping = cor2_dist.idxmax()
            else:
                preservation = f"POOR ({max_pct:.1f}%)"
                main_mapping = "mixed"
        
        print(f"  {cor_val}: {preservation} -> mainly COR 2 = {main_mapping}")
        for cor2_val, count in cor2_dist.items():
            pct = (count / len(subset)) * 100
            print(f"    COR 2 = {cor2_val}: {count} records ({pct:.1f}%)")
    print()
    
    # Final detailed recommendation
    print("=== DETAILED RECOMMENDATION ===")
    
    print("ANALYSIS SUMMARY:")
    print("1. COR 2 appears to be a BINARY RECODING of the original COR variable")
    print("2. The recoding seems to group:")
    print("   - 'BRANCO' (White) -> 0 (with 1 exception in 'OUTRAS')")
    print("   - 'NEGRA' (Black) + 'PARDA' (Mixed) + most 'OUTRAS' -> 1")
    print("3. This creates a minority/non-minority binary classification")
    print()
    
    print("INFORMATION CONTENT:")
    print("- COR preserves all original racial/ethnic distinctions (4 categories)")
    print("- COR 2 reduces this to a binary minority classification (2 categories)")
    print("- Information is LOST in the transformation (4 -> 2 categories)")
    print()
    
    print("MODELING RECOMMENDATIONS:")
    print()
    print("OPTION 1 - USE BOTH (Recommended for most cases):")
    print("  ✓ COR provides granular racial/ethnic information")
    print("  ✓ COR 2 provides simplified minority/majority classification")
    print("  ✓ They capture different aspects of the same underlying construct")
    print("  ✓ May be valuable for different types of analysis")
    print("  ⚠ Need to check for multicollinearity in final model")
    print()
    
    print("OPTION 2 - USE ONLY COR:")
    print("  ✓ Preserves all original information")
    print("  ✓ More detailed for healthcare disparities analysis")
    print("  ⚠ May have small sample sizes for some categories (OUTRAS: n=7)")
    print()
    
    print("OPTION 3 - USE ONLY COR 2:")
    print("  ✓ Simpler binary classification")
    print("  ✓ Better statistical power with balanced groups")
    print("  ✗ Loses important distinctions (e.g., Black vs Mixed)")
    print("  ✗ Less informative for healthcare research")
    print()
    
    print("FINAL RECOMMENDATION:")
    print("For healthcare/medical research, KEEP BOTH variables but use them strategically:")
    print("- Use COR for detailed racial/ethnic analysis")
    print("- Use COR 2 for minority/majority comparisons")
    print("- Consider creating dummy variables from COR for regression models")
    print("- Monitor for multicollinearity and remove one if VIF > 5-10")
    
    return df, crosstab

if __name__ == "__main__":
    df, crosstab = detailed_cor_analysis()