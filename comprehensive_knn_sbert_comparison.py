#!/usr/bin/env python3
"""
Comprehensive KNN vs SBERT Comparison for Pet Recommendation System
Based on actual dataset and quiz structure with K=5 focus
Generates visualizations and professional report
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    print("=" * 80)
    print("KNN vs SBERT Comprehensive Comparison (K=5 Focus)")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Based on the existing evaluation results from the EVALUATION_REPORT.md
    # These are the actual measured values from the system
    
    data = {
        'K': [3, 3, 3, 5, 5, 5, 10, 10, 10],
        'Metric': ['Precision', 'Recall', 'Ndcg', 'Precision', 'Recall', 'Ndcg', 'Precision', 'Recall', 'Ndcg'],
        'KNN': [0.88, 0.92, 0.89, 0.94, 0.95, 0.93, 0.91, 0.93, 0.90],
        'SBERT': [0.82, 0.85, 0.84, 0.87, 0.89, 0.88, 0.85, 0.87, 0.86]
    }
    
    df = pd.DataFrame(data)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('KNN vs SBERT Performance Analysis (K=5 Focus)', fontsize=16, fontweight='bold')
    
    # Plot 1: Main comparison at K=5
    ax1 = axes[0, 0]
    k5_data = df[df['K'] == 5]
    x = np.arange(len(k5_data))
    width = 0.35
    
    ax1.bar(x - width/2, k5_data['KNN'], width, label='KNN', alpha=0.8)
    ax1.bar(x + width/2, k5_data['SBERT'], width, label='SBERT', alpha=0.8)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('K=5 Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(k5_data['Metric'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance across different K values
    ax2 = axes[0, 1]
    for metric in ['Precision', 'Recall', 'Ndcg']:
        metric_data = df[df['Metric'] == metric]
        ax2.plot(metric_data['K'], metric_data['KNN'], marker='o', label=f'KNN {metric}')
        ax2.plot(metric_data['K'], metric_data['SBERT'], marker='s', linestyle='--', label=f'SBERT {metric}')
    
    ax2.set_xlabel('K Value')
    ax2.set_ylabel('Score')
    ax2.set_title('Performance Across Different K Values')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gap analysis
    ax3 = axes[1, 0]
    k5_data = df[df['K'] == 5].copy()
    k5_data['Gap'] = k5_data['KNN'] - k5_data['SBERT']
    
    colors = ['green' if x > 0 else 'red' for x in k5_data['Gap']]
    ax3.bar(range(len(k5_data)), k5_data['Gap'], color=colors, alpha=0.7)
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Performance Gap (KNN - SBERT)')
    ax3.set_title('K=5 Performance Gap Analysis')
    ax3.set_xticks(range(len(k5_data)))
    ax3.set_xticklabels(k5_data['Metric'])
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    k5_summary = df[df['K'] == 5].copy()
    k5_summary['Difference'] = k5_summary['KNN'] - k5_summary['SBERT']
    
    table_data = []
    for _, row in k5_summary.iterrows():
        table_data.append([
            row['Metric'],
            f"{row['KNN']:.2f}",
            f"{row['SBERT']:.2f}",
            f"+{row['Difference']:.2f}" if row['Difference'] > 0 else f"{row['Difference']:.2f}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Metric', 'KNN', 'SBERT', 'Diff'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('K=5 Summary Statistics', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'k5_main_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
