#!/usr/bin/env python3
"""
Analysis script to examine the relationship between COR and COR 2 columns
in the DRC dataset to determine feature redundancy.
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_cor_relationship():
    """
    Analyze the relationship between COR and COR 2 columns in the DRC dataset.
    """
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('/Users/aiacontext/PycharmProjects/jaliRNA/dataset/dataset_drc_processed.csv', sep=';')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total records: {len(df)}")
    print()
    
    # Basic information about the columns
    print("=== BASIC COLUMN INFORMATION ===")
    print(f"COR column:")
    print(f"  - Data type: {df['COR'].dtype}")
    print(f"  - Non-null count: {df['COR'].count()}")
    print(f"  - Null count: {df['COR'].isnull().sum()}")
    
    print(f"\nCOR 2 column:")
    print(f"  - Data type: {df['COR 2'].dtype}")
    print(f"  - Non-null count: {df['COR 2'].count()}")
    print(f"  - Null count: {df['COR 2'].isnull().sum()}")
    print()
    
    # Examine unique values
    print("=== UNIQUE VALUES ANALYSIS ===")
    cor_unique = df['COR'].unique()
    cor2_unique = df['COR 2'].unique()
    
    print("COR unique values:")
    for val in sorted(cor_unique):
        count = (df['COR'] == val).sum()
        percentage = (count / len(df)) * 100
        print(f"  '{val}': {count} records ({percentage:.1f}%)")
    
    print(f"\nCOR 2 unique values:")
    for val in sorted(cor2_unique):
        count = (df['COR 2'] == val).sum()
        percentage = (count / len(df)) * 100
        print(f"  {val}: {count} records ({percentage:.1f}%)")
    print()
    
    # Create crosstab to examine relationship
    print("=== CROSSTAB ANALYSIS ===")
    crosstab = pd.crosstab(df['COR'], df['COR 2'], margins=True)
    print("Crosstab (COR vs COR 2):")
    print(crosstab)
    print()
    
    # Calculate correlation if both are numeric or can be converted
    print("=== CORRELATION ANALYSIS ===")
    try:
        # Convert COR 2 to numeric if it's not already
        cor2_numeric = pd.to_numeric(df['COR 2'], errors='coerce')
        
        # Try to create a mapping for COR to numeric values
        cor_mapping = {}
        for i, val in enumerate(sorted(df['COR'].unique())):
            cor_mapping[val] = i
        
        cor_numeric = df['COR'].map(cor_mapping)
        
        correlation = cor_numeric.corr(cor2_numeric)
        print(f"Correlation between COR (mapped to numeric) and COR 2: {correlation:.4f}")
        
    except Exception as e:
        print(f"Could not calculate correlation: {e}")
    print()
    
    # Analyze the mapping pattern
    print("=== MAPPING PATTERN ANALYSIS ===")
    mapping_df = df.groupby(['COR', 'COR 2']).size().reset_index(name='count')
    mapping_df = mapping_df.sort_values(['COR', 'COR 2'])
    
    print("COR to COR 2 mapping:")
    for _, row in mapping_df.iterrows():
        print(f"  {row['COR']} -> {row['COR 2']}: {row['count']} records")
    print()
    
    # Check for perfect mapping (one-to-one relationship)
    print("=== REDUNDANCY ANALYSIS ===")
    
    # Check if there's a perfect one-to-one mapping
    cor_to_cor2_mapping = {}
    is_perfect_mapping = True
    
    for cor_val in df['COR'].unique():
        cor2_vals = df[df['COR'] == cor_val]['COR 2'].unique()
        cor_to_cor2_mapping[cor_val] = cor2_vals
        
        if len(cor2_vals) > 1:
            is_perfect_mapping = False
            print(f"COR '{cor_val}' maps to multiple COR 2 values: {cor2_vals}")
    
    if is_perfect_mapping:
        print("Perfect one-to-one mapping detected between COR and COR 2")
        print("COR 2 appears to be a encoded/transformed version of COR")
    else:
        print("No perfect one-to-one mapping - COR 2 may provide additional information")
    
    # Check reverse mapping
    cor2_to_cor_mapping = {}
    reverse_perfect_mapping = True
    
    for cor2_val in df['COR 2'].unique():
        cor_vals = df[df['COR 2'] == cor2_val]['COR'].unique()
        cor2_to_cor_mapping[cor2_val] = cor_vals
        
        if len(cor_vals) > 1:
            reverse_perfect_mapping = False
            print(f"COR 2 '{cor2_val}' maps to multiple COR values: {cor_vals}")
    
    print()
    
    # Information content analysis
    print("=== INFORMATION CONTENT ANALYSIS ===")
    cor_entropy = -sum((df['COR'].value_counts(normalize=True) * np.log2(df['COR'].value_counts(normalize=True))))
    cor2_entropy = -sum((df['COR 2'].value_counts(normalize=True) * np.log2(df['COR 2'].value_counts(normalize=True))))
    
    print(f"COR entropy (information content): {cor_entropy:.4f}")
    print(f"COR 2 entropy (information content): {cor2_entropy:.4f}")
    
    if abs(cor_entropy - cor2_entropy) < 0.001:
        print("Similar entropy suggests similar information content")
    else:
        print("Different entropy suggests different information content")
    print()
    
    # Final recommendation
    print("=== RECOMMENDATION ===")
    
    if is_perfect_mapping and reverse_perfect_mapping:
        print("RECOMMENDATION: COR 2 is REDUNDANT")
        print("Reasons:")
        print("- Perfect bidirectional mapping between COR and COR 2")
        print("- COR 2 appears to be just a numeric encoding of COR")
        print("- Including both would introduce multicollinearity")
        print("- Recommend using only one of them (preferably the numeric COR 2 for modeling)")
        
    elif correlation is not None and abs(correlation) > 0.9:
        print("RECOMMENDATION: COR 2 is LIKELY REDUNDANT")
        print("Reasons:")
        print(f"- High correlation ({correlation:.4f}) suggests strong linear relationship")
        print("- May introduce multicollinearity issues")
        print("- Consider using only one for modeling")
        
    else:
        print("RECOMMENDATION: COR 2 provides ADDITIONAL INFORMATION")
        print("Reasons:")
        print("- No perfect mapping between COR and COR 2")
        print("- Different information content")
        print("- Both columns may be valuable for modeling")
        print("- Consider including both as separate features")
    
    return df, crosstab, mapping_df

if __name__ == "__main__":
    df, crosstab, mapping_df = analyze_cor_relationship()