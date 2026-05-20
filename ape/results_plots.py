import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path: str = 'comparison.csv') -> pd.DataFrame:
    """Load and prepare the comparison data"""
    df = pd.read_csv(csv_path)
    df['Method_Type'] = df['Method'].apply(lambda x:
        'APE' if '-APE' in x
        else 'TADILER' if '-TADILER' in x
        else 'Original'
    )
    return df


def plot_performance_comparison_strategies(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """Performance comparison across strategies - individual plot"""

    fig, ax = plt.subplots(figsize=figsize)

    colors = {'Original': '#d62728', 'TADILER': '#ff7f0e', 'APE': '#2ca02c'}

    strategies = sorted(df['Strategy'].unique())

    method_means = {}
    method_stds = {}

    for method in ['Original', 'TADILER', 'APE']:
        means = []
        stds = []
        for strategy in strategies:
            subset = df[(df['Strategy'] == strategy) & (df['Method_Type'] == method)]
            if not subset.empty:
                means.append(subset['Mean AMCA'].mean())
                stds.append(subset['Mean AMCA'].std())
            else:
                means.append(0)
                stds.append(0)
        method_means[method] = means
        method_stds[method] = stds

    x = np.arange(len(strategies))
    width = 0.25

    for i, method in enumerate(['Original', 'TADILER', 'APE']):
        bars = ax.bar(x + i*width, method_means[method], width,
                     label=method, color=colors[method], alpha=0.8,
                     yerr=method_stds[method], capsize=4, edgecolor='black', linewidth=1.2)
        for bar, mean_val in zip(bars, method_means[method]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xlabel('Continual learning strategy', fontweight='bold', fontsize=14)
    ax.set_ylabel('Mean AMCA', fontweight='bold', fontsize=14)
    ax.set_title('Performance comparison across strategies', fontweight='bold', fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies, fontweight='bold', fontsize=12)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.75, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Performance comparison plot saved to {save_path}")

    plt.show()


def plot_amca_vs_forgetting(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """AMCA vs forgetting trade-off scatter plot"""

    fig, ax = plt.subplots(figsize=figsize)

    colors = {'Original': '#d62728', 'TADILER': '#ff7f0e', 'APE': '#2ca02c'}
    markers = {'Original': 'o', 'TADILER': 's', 'APE': '^'}
    sizes = {'Original': 60, 'TADILER': 70, 'APE': 80}

    for method in ['Original', 'TADILER', 'APE']:
        method_data = df[df['Method_Type'] == method]
        if not method_data.empty:
            ax.scatter(abs(method_data['Forgetting']), method_data['Mean AMCA'],
                       label=method, color=colors[method], marker=markers[method],
                       s=sizes[method], alpha=0.7, edgecolors='black', linewidth=1.0)

    ax.set_xlabel('Absolute forgetting rate', fontweight='bold', fontsize=14)
    ax.set_ylabel('Mean AMCA', fontweight='bold', fontsize=14)
    ax.set_title('AMCA vs forgetting trade-off', fontweight='bold', fontsize=16)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')

    ax.annotate('Ideal region:\nHigh AMCA, Low forgetting',
                xy=(0.005, 0.96), xytext=(0.02, 0.94),
                fontsize=11, ha='left', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"AMCA vs forgetting plot saved to {save_path}")

    plt.show()


def plot_performance_evolution(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """Performance evolution across KNN values"""

    fig, ax = plt.subplots(figsize=figsize)

    colors = {'Original': '#d62728', 'TADILER': '#ff7f0e', 'APE': '#2ca02c'}
    linestyles = {'Original': '-', 'TADILER': '--', 'APE': '-'}
    markers = {'Original': 'o', 'TADILER': 's', 'APE': '^'}

    knn_values = sorted(df['KNN'].unique())

    for method in ['Original', 'TADILER', 'APE']:
        means = []
        stds = []
        for knn in knn_values:
            subset = df[(df['KNN'] == knn) & (df['Method_Type'] == method)]
            if not subset.empty:
                means.append(subset['Mean AMCA'].mean())
                stds.append(subset['Mean AMCA'].std())
            else:
                means.append(0)
                stds.append(0)

        ax.plot(knn_values, means, label=method, color=colors[method],
               linestyle=linestyles[method], marker=markers[method],
               linewidth=3, markersize=8, markeredgecolor='black', markeredgewidth=1)

        means_array = np.array(means)
        stds_array = np.array(stds)
        ax.fill_between(knn_values, means_array - stds_array, means_array + stds_array,
                       color=colors[method], alpha=0.2)

    ax.set_xlabel('Number of nearest neighbors (KNN)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Mean AMCA', fontweight='bold', fontsize=14)
    ax.set_title('Performance evolution across KNN values', fontweight='bold', fontsize=16)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0.85, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Performance evolution plot saved to {save_path}")

    plt.show()


def plot_single_architecture_radar(
    df: pd.DataFrame,
    architecture: str = 'Attention',
    metric: str = 'Mean AMCA',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """Create a single radar plot for one architecture with strategies as axes"""

    arch_df = df[df['Architecture'] == architecture]
    strategies = sorted(arch_df['Strategy'].unique())

    colors = {'Original': '#d62728', 'TADILER': '#ff7f0e', 'APE': '#2ca02c'}

    N = len(strategies)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for method_type in ['Original', 'TADILER', 'APE']:
        values = []
        for strategy in strategies:
            strategy_method_df = arch_df[
                (arch_df['Strategy'] == strategy) &
                (arch_df['Method_Type'] == method_type)
            ]
            if not strategy_method_df.empty:
                values.append(strategy_method_df[metric].mean())
            else:
                values.append(0)

        values += [values[0]]
        ax.plot(angles, values, label=method_type, linewidth=3,
               color=colors[method_type], linestyle='solid')
        ax.fill(angles, values, alpha=0.15, color=colors[method_type])

    ax.set_thetagrids(np.degrees(angles[:-1]), strategies, fontsize=12, fontweight='bold')
    ax.set_title(f'{architecture} architecture performance',
                fontweight='bold', pad=20, fontsize=16)

    if 'AMCA' in metric:
        ax.set_ylim(0, 1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Radar plot saved to {save_path}")

    plt.show()


def plot_overall_radar(
    df: pd.DataFrame,
    metric: str = 'Mean AMCA',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """Create a radar plot averaged across all architectures with strategies as axes"""

    strategies = sorted(df['Strategy'].unique())
    colors = {'Original': '#d62728', 'TADILER': '#ff7f0e', 'APE': '#2ca02c'}

    N = len(strategies)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for method_type in ['Original', 'TADILER', 'APE']:
        values = []
        for strategy in strategies:
            strategy_method_df = df[
                (df['Strategy'] == strategy) &
                (df['Method_Type'] == method_type)
            ]
            if not strategy_method_df.empty:
                values.append(strategy_method_df[metric].mean())
            else:
                values.append(0)

        values += [values[0]]
        ax.plot(angles, values, label=method_type, linewidth=3,
               color=colors[method_type], linestyle='solid')
        ax.fill(angles, values, alpha=0.15, color=colors[method_type])

    ax.set_thetagrids(np.degrees(angles[:-1]), strategies, fontsize=12, fontweight='bold')
    ax.set_title('Overall performance across all architectures',
                fontweight='bold', pad=20, fontsize=16)

    if 'AMCA' in metric:
        ax.set_ylim(0, 1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Radar plot saved to {save_path}")

    plt.show()


def create_journal_plots(csv_path: str = 'comparison.csv'):
    """Create all plots for journal submission"""
    plt.style.use('default')
    sns.set_palette("husl")

    df = load_and_prepare_data(csv_path)

    print("Creating plots for journal submission:")

    print("\n1. Performance comparison across strategies:")
    plot_performance_comparison_strategies(df, save_path='performance_strategies.pdf')

    print("\n2. AMCA vs forgetting trade-off:")
    plot_amca_vs_forgetting(df, save_path='amca_forgetting.pdf')

    print("\n3. Performance evolution across KNN values:")
    plot_performance_evolution(df, save_path='performance_evolution.pdf')

    plot_overall_radar(df, save_path='radar_overall.pdf')
