#!/usr/bin/env python3
"""
Comprehensive KNN vs SBERT Comparison for Pet Recommendation System
Generates visualizations and detailed analysis focusing on K=5
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("KNN vs SBERT Comprehensive Analysis (K=5)")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}, Pandas: {pd.__version__}, Matplotlib: {matplotlib.__version__}")
    print()

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Create output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    # Sample data - replace with actual evaluation results
    data = {
        'Metric': ['Precision@5', 'Recall@5', 'NDCG@5', 'Coverage', 'Diversity'],
        'KNN': [0.94, 0.95, 0.93, 0.89, 0.76],
        'SBERT': [0.87, 0.89, 0.88, 0.92, 0.81],
        'KNN_std': [0.05, 0.04, 0.06, 0.07, 0.08],
        'SBERT_std': [0.06, 0.05, 0.07, 0.06, 0.09]
    }
    
    df = pd.DataFrame(data)
    
    # Create visualizations
    create_main_comparison(df, output_dir)
    create_radar_chart(df, output_dir)
    create_gap_analysis(df, output_dir)
    create_summary_table(df, output_dir)
    create_comprehensive_table(df, output_dir)
    
    # Save CSV
    csv_file = output_dir / 'results_table.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV: {csv_file}")
    
    print("\n" + "=" * 80)
    print("✅ All visualizations generated successfully!")
    print("=" * 80)

def create_main_comparison(df, output_dir):
    """Main bar chart comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['KNN'], width, label='KNN', 
                   yerr=df['KNN_std'], capsize=5, alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, df['SBERT'], width, label='SBERT',
                   yerr=df['SBERT_std'], capsize=5, alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('KNN vs SBERT Performance Comparison (K=5)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Metric'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / 'k5_main_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()

def create_radar_chart(df, output_dir):
    """Radar chart for performance comparison"""
    metrics = df['Metric'].tolist()
    knn_values = df['KNN'].tolist()
    sbert_values = df['SBERT'].tolist()
    
    # Number of variables
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    knn_values += knn_values[:1]
    sbert_values += sbert_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, knn_values, 'o-', linewidth=2, label='KNN', color='#3498db')
    ax.fill(angles, knn_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, sbert_values, 'o-', linewidth=2, label='SBERT', color='#e74c3c')
    ax.fill(angles, sbert_values, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=11)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart (K=5)', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    output_file = output_dir / 'k5_radar_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()

def create_gap_analysis(df, output_dir):
    """Performance gap analysis"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df['Gap'] = df['KNN'] - df['SBERT']
    colors = ['green' if x > 0 else 'red' for x in df['Gap']]
    
    bars = ax.bar(range(len(df)), df['Gap'], color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Gap (KNN - SBERT)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Gap Analysis (K=5)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Metric'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.3f}',
               ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'k5_gap_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()

def create_summary_table(df, output_dir):
    """Summary statistics table"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        gap = row['KNN'] - row['SBERT']
        table_data.append([
            row['Metric'],
            f"{row['KNN']:.4f} ± {row['KNN_std']:.4f}",
            f"{row['SBERT']:.4f} ± {row['SBERT_std']:.4f}",
            f"{gap:+.4f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'KNN', 'SBERT', 'Gap'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Summary Statistics (K=5)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = output_dir / 'k5_summary_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()

def create_comprehensive_table(df, output_dir):
    """Comprehensive comparison table"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Extended table with more details
    table_data = []
    for _, row in df.iterrows():
        gap = row['KNN'] - row['SBERT']
        pct_diff = (gap / row['SBERT']) * 100 if row['SBERT'] > 0 else 0
        winner = 'KNN' if gap > 0 else 'SBERT' if gap < 0 else 'Tie'
        
        table_data.append([
            row['Metric'],
            f"{row['KNN']:.4f}",
            f"{row['KNN_std']:.4f}",
            f"{row['SBERT']:.4f}",
            f"{row['SBERT_std']:.4f}",
            f"{gap:+.4f}",
            f"{pct_diff:+.1f}%",
            winner
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'KNN', 'KNN Std', 'SBERT', 'SBERT Std', 'Gap', '%  Diff', 'Winner'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(8):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code rows
    for i in range(1, len(table_data) + 1):
        for j in range(8):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
        
        # Highlight winner column
        winner = table_data[i-1][7]
        if winner == 'KNN':
            table[(i, 7)].set_facecolor('#d5f4e6')
        elif winner == 'SBERT':
            table[(i, 7)].set_facecolor('#fadbd8')
    
    ax.set_title('Comprehensive Performance Comparison (K=5)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = output_dir / 'k5_comprehensive_table.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
