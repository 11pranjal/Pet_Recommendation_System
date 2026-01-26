#!/usr/bin/env python3
"""
Visualization script for SBERT evaluation results
Creates comparison charts and summary report
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def create_visualizations():
    """Create comprehensive visualization of SBERT results"""
    
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    results_file = output_dir / "sbert_new_dataset_results.csv"
    df = pd.read_csv(results_file)
    
    print("="*80)
    print("📊 CREATING VISUALIZATIONS FOR SBERT RESULTS")
    print("="*80)
    print(f"\nLoaded results from: {results_file}")
    print(df.to_string(index=False))
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    
    # Main grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: K=5 Performance (Main)
    ax1 = fig.add_subplot(gs[0, :2])
    k5_data = df[df['K'] == 5].iloc[0]
    metrics = ['Precision', 'Recall', 'NDCG']
    values = [k5_data['Precision'], k5_data['Recall'], k5_data['NDCG']]
    
    bars = ax1.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('SBERT Performance at K=5 (New Dataset)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Summary Statistics Box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    summary_text = f"""
    📊 K=5 PERFORMANCE
    
    Precision:  {k5_data['Precision']:.2%}
    Recall:     {k5_data['Recall']:.2%}
    NDCG:       {k5_data['NDCG']:.2%}
    
    Avg Similarity: {k5_data['Avg_Similarity']:.3f}
    
    ━━━━━━━━━━━━━━━━
    Dataset: 1,985 pets
    Dimensions: 384
    Model: MiniLM-L6-v2
    """
    
    ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            family='monospace')
    
    # Plot 3: Performance Across K Values
    ax3 = fig.add_subplot(gs[1, :])
    
    x = df['K']
    width = 0.25
    x_pos = np.arange(len(x))
    
    ax3.bar(x_pos - width, df['Precision'], width, label='Precision', alpha=0.8, color='#FF6B6B')
    ax3.bar(x_pos, df['Recall'], width, label='Recall', alpha=0.8, color='#4ECDC4')
    ax3.bar(x_pos + width, df['NDCG'], width, label='NDCG', alpha=0.8, color='#45B7D1')
    
    ax3.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Metrics Across Different K Values', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df['K'])
    ax3.legend(fontsize=10, loc='lower right')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Precision Trend
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df['K'], df['Precision'], marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    ax4.fill_between(df['K'], df['Precision'], alpha=0.3, color='#FF6B6B')
    ax4.set_xlabel('K', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax4.set_title('Precision vs K', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.75, 0.90])
    
    # Plot 5: Recall Trend
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(df['K'], df['Recall'], marker='s', linewidth=2, markersize=8, color='#4ECDC4')
    ax5.fill_between(df['K'], df['Recall'], alpha=0.3, color='#4ECDC4')
    ax5.set_xlabel('K', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax5.set_title('Recall vs K', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0.75, 0.90])
    
    # Plot 6: NDCG Trend
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(df['K'], df['NDCG'], marker='^', linewidth=2, markersize=8, color='#45B7D1')
    ax6.fill_between(df['K'], df['NDCG'], alpha=0.3, color='#45B7D1')
    ax6.set_xlabel('K', fontsize=11, fontweight='bold')
    ax6.set_ylabel('NDCG', fontsize=11, fontweight='bold')
    ax6.set_title('NDCG vs K', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0.75, 0.95])
    
    # Add main title
    fig.suptitle('SBERT Model Performance Analysis - New Dataset\n(sbert_refined_data_with_breed_characteristics_gender_full.csv)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_file = output_dir / 'sbert_new_dataset_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_file}")
    
    plt.close()
    
    # Create comparison table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            f"K = {int(row['K'])}",
            f"{row['Precision']:.2%}",
            f"{row['Recall']:.2%}",
            f"{row['NDCG']:.2%}",
            f"{row['Avg_Similarity']:.4f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['K Value', 'Precision', 'Recall', 'NDCG', 'Avg Similarity'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white', size=13)
    
    # Style cells - highlight K=5
    for i in range(1, 4):
        for j in range(5):
            if i == 2:  # K=5 row
                table[(i, j)].set_facecolor('#E8F8F5')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#F8F9F9' if i % 2 == 0 else 'white')
    
    plt.title('SBERT Performance Summary - New Dataset\nDetailed Results Across Different K Values',
             fontsize=16, fontweight='bold', pad=20)
    
    table_file = output_dir / 'sbert_new_dataset_table.png'
    plt.savefig(table_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved table: {table_file}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("✅ VISUALIZATIONS CREATED SUCCESSFULLY")
    print("="*80)
    print("\nFiles created:")
    print(f"  1. {output_file}")
    print(f"  2. {table_file}")
    print(f"  3. {results_file}")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("📈 SBERT RESULTS VISUALIZATION")
    print("="*80)
    
    create_visualizations()
    
    print("\n✨ Complete! Check the evaluation_results/ directory for visualizations.")

if __name__ == "__main__":
    main()
