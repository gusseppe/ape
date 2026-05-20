import matplotlib.pyplot as plt
import numpy as np
import os


def _setup_plot_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


def plot_comprehensive_performance_analysis(best_values):
    """Performance comparison: learning vs transfer across tasks."""
    tasks = [k for k in best_values.keys() if isinstance(k, int)]
    ape_scores = [best_values[task]['best_score'] for task in tasks]

    if 'ape_tracking' in best_values:
        baseline_scores = best_values['ape_tracking']['summary'].get('baseline_scores', {})
        exhaustive_scores = []
        for task in tasks:
            if task == 0:
                exhaustive_scores.append(baseline_scores.get('exhaustive_search', 0.8921))
            else:
                exhaustive_scores.append(baseline_scores.get(f'exhaustive_task{task}', 0.89))
    else:
        exhaustive_scores = [0.8921, 0.8899, 0.8909][:len(tasks)]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(tasks))
    width = 0.35

    ape_colors = ['#228B22'] + ['#90EE90'] * (len(tasks) - 1)
    exhaustive_colors = ['#B22222'] + ['#FFA07A'] * (len(tasks) - 1)

    bars1 = ax.bar(x - width/2, ape_scores, width, label='APE',
                   color=ape_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, exhaustive_scores, width, label='Exhaustive search',
                   color=exhaustive_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    for i, (ape_score, exhaustive_score) in enumerate(zip(ape_scores, exhaustive_scores)):
        diff = ape_score - exhaustive_score
        color = 'green' if diff > 0 else 'red'
        symbol = '+' if diff > 0 else ''
        ax.annotate(f'{symbol}{diff:.3f}',
                   xy=(i, max(ape_score, exhaustive_score) + 0.015),
                   ha='center', va='bottom', fontweight='bold', color=color, fontsize=9)

    ax.set_xlabel('Evolution iteration')
    ax.set_ylabel('F1-score')
    ax.set_title('Performance comparison: learning vs transfer across tasks')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{"(Learning)" if i == 0 else "(Transfer)"}' for i in tasks])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs('extension_plots', exist_ok=True)
    plt.savefig('extension_plots/performance_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_computational_efficiency(best_values):
    """Computational efficiency comparison."""
    if 'ape_tracking' in best_values:
        ape_evaluations = best_values['ape_tracking']['summary']['total_candidates']
    else:
        ape_evaluations = 25

    exhaustive_evaluations = 110

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['APE', 'Exhaustive search']
    evaluations = [ape_evaluations, exhaustive_evaluations]
    colors = ['#32CD32', '#FF6347']

    bars = ax.bar(methods, evaluations, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1, width=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    efficiency_gain = exhaustive_evaluations / ape_evaluations
    ax.text(0.5, max(evaluations) * 0.5, f'{efficiency_gain:.1f}× more\nefficient',
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFF99', alpha=0.9,
                      edgecolor='black', linewidth=1))

    ax.set_ylabel('Number of evaluations')
    ax.set_title('Computational efficiency comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(evaluations) * 1.1)

    plt.tight_layout()
    os.makedirs('extension_plots', exist_ok=True)
    plt.savefig('extension_plots/computational_efficiency.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_progression(best_values):
    """Learning progression: start vs final stages."""
    if 'ape_tracking' not in best_values:
        print("❌ No tracking data available for learning progression")
        return

    tracking_data = best_values['ape_tracking']
    evolution_history = tracking_data['evolution_history']
    summary = tracking_data['summary']

    start_state = evolution_history[0]
    end_state = evolution_history[-1]

    fig, ax = plt.subplots(figsize=(12, 6))

    stages = ['Start', 'Final']
    f1_scores = [start_state['f1'], end_state['f1']]
    colors = ['#FF6B6B', '#4ECDC4']

    bars = ax.bar(stages, f1_scores, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1, width=0.5)

    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.annotate(f'F1: {height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                   fontsize=12, fontweight='bold')

    improvement = summary['total_improvement']
    ax.text(0.5, (f1_scores[0] + f1_scores[1])/2,
            f'Improvement\n+{improvement:.3f}\nin {summary["evolution_steps"]} steps',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8,
                     edgecolor='black', linewidth=1))

    start_prompt = f"Template: '{start_state['template']}'\n" if start_state['template'] else "Template: (empty)\n"
    start_prompt += f"Healthy: '{start_state['descriptions']['No retinopathy']}'\n"
    start_prompt += f"Disease: '{start_state['descriptions']['Retinopathy']}'"

    end_prompt = f"Template: '{end_state['template']}'\n"
    end_prompt += f"Healthy: '{end_state['descriptions']['No retinopathy']}'\n"
    end_prompt += f"Disease: '{end_state['descriptions']['Retinopathy']}'"

    ax.text(0, -0.15, start_prompt, ha='center', va='top', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
           transform=ax.get_xaxis_transform())

    ax.text(1, -0.15, end_prompt, ha='center', va='top', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
           transform=ax.get_xaxis_transform())

    stats_text = f"Evolution time: {summary['total_time']:.1f}s\n"
    stats_text += f"Total candidates: {summary['total_candidates']}"

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_ylabel('F1-score')
    ax.set_title('APE autonomous learning: from basic to optimal prompts')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(f1_scores) * 1.4)

    plt.tight_layout()
    os.makedirs('extension_plots', exist_ok=True)
    plt.savefig('extension_plots/learning_progression.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_candidate_quality_distribution(best_values):
    """Candidate quality distribution across iterations."""
    if 'ape_tracking' not in best_values:
        print("❌ No tracking data available for candidate quality distribution")
        return

    tracking_data = best_values['ape_tracking']
    all_candidates = tracking_data['all_candidates']
    evolution_history = tracking_data['evolution_history']

    if not all_candidates:
        print("❌ No candidate data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = sorted(set(candidate['iteration'] for candidate in all_candidates))
    iteration_data = []
    iteration_labels = []

    for iteration in iterations:
        iter_candidates = [c for c in all_candidates if c['iteration'] == iteration]
        f1_scores = [c['f1'] for c in iter_candidates]
        iteration_data.append(f1_scores)
        iteration_labels.append(f'{iteration}')

    parts = ax.violinplot(iteration_data, positions=range(len(iterations)),
                         showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#87CEEB')
        pc.set_alpha(0.7)

    selected_f1s = []
    selected_positions = []

    for i, iteration in enumerate(iterations):
        for step in evolution_history[1:]:
            if step['iteration'] == iteration:
                selected_f1s.append(step['f1'])
                selected_positions.append(i)
                break

    if selected_positions:
        ax.scatter(selected_positions, selected_f1s, c='#FFD700', s=100, marker='*',
                  edgecolors='black', linewidth=1.5, label='Selected for evolution', zorder=5)

    if len(selected_positions) > 1:
        z = np.polyfit(selected_positions, selected_f1s, 1)
        p = np.poly1d(z)
        ax.plot(selected_positions, p(selected_positions), 'r--', alpha=0.8,
               linewidth=2, label='Evolution trend')

    ax.set_xlabel('Evolution iteration')
    ax.set_ylabel('F1-score distribution')
    ax.set_title('Candidate quality distribution across iterations')
    ax.set_xticks(range(len(iterations)))
    ax.set_xticklabels(iteration_labels)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs('extension_plots', exist_ok=True)
    plt.savefig('extension_plots/candidate_quality_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_search_space_exploration(best_values):
    """Search space exploration: APE navigation through prompt landscape."""
    if 'ape_tracking' not in best_values:
        print("❌ No tracking data available for search space exploration")
        return

    tracking_data = best_values['ape_tracking']
    all_candidates = tracking_data['all_candidates']
    evolution_history = tracking_data['evolution_history']

    if not all_candidates:
        print("❌ No candidate data available")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    all_templates = set()
    description_pairs = set()

    for candidate in all_candidates:
        all_templates.add(candidate['template'] if candidate['template'] else '(empty)')
        healthy = candidate['descriptions']['No retinopathy']
        disease = candidate['descriptions']['Retinopathy']
        description_pairs.add((healthy, disease))

    template_list = sorted(list(all_templates))
    description_list = sorted(list(description_pairs))

    x_coords, y_coords, f1_scores, is_selected = [], [], [], []

    for candidate in all_candidates:
        template = candidate['template'] if candidate['template'] else '(empty)'
        healthy = candidate['descriptions']['No retinopathy']
        disease = candidate['descriptions']['Retinopathy']
        desc_pair = (healthy, disease)

        x_coords.append(template_list.index(template))
        y_coords.append(description_list.index(desc_pair))
        f1_scores.append(candidate['f1'])

        selected = any(
            step['template'] == candidate['template'] and
            step['descriptions'] == candidate['descriptions']
            for step in evolution_history[1:]
        )
        is_selected.append(selected)

    scatter = ax.scatter(x_coords, y_coords, c=f1_scores, s=60,
                        cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=0.5)

    selected_x = [x for x, s in zip(x_coords, is_selected) if s]
    selected_y = [y for y, s in zip(y_coords, is_selected) if s]

    if selected_x:
        ax.scatter(selected_x, selected_y, s=200, marker='*', c='gold',
                  edgecolors='black', linewidth=2, label='Selected for evolution', zorder=5)
        if len(selected_x) > 1:
            ax.plot(selected_x, selected_y, 'r-', linewidth=2, alpha=0.8,
                   label='Evolution path', zorder=4)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1-score')

    ax.set_xlabel('Template variety')
    ax.set_ylabel('Description pair variety')
    ax.set_title('Search space exploration: APE navigation through prompt landscape')

    max_labels = 8
    if len(template_list) > max_labels:
        step = len(template_list) // max_labels
        ax.set_xticks(range(0, len(template_list), step))
        ax.set_xticklabels([template_list[i][:15] + '...' if len(template_list[i]) > 15
                           else template_list[i] for i in range(0, len(template_list), step)],
                          rotation=45, ha='right')
    else:
        ax.set_xticks(range(len(template_list)))
        ax.set_xticklabels([t[:15] + '...' if len(t) > 15 else t for t in template_list],
                          rotation=45, ha='right')

    if len(description_list) > max_labels:
        step = len(description_list) // max_labels
        ax.set_yticks(range(0, len(description_list), step))
        ax.set_yticklabels([f"{desc[0][:10]}...\nvs\n{desc[1][:10]}..."
                           for desc in [description_list[i] for i in range(0, len(description_list), step)]],
                          fontsize=8)
    else:
        ax.set_yticks(range(len(description_list)))
        ax.set_yticklabels([f"{desc[0][:10]}...\nvs\n{desc[1][:10]}..."
                           for desc in description_list], fontsize=8)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    exploration_text = f"Templates explored: {len(all_templates)}\n"
    exploration_text += f"Description pairs: {len(description_pairs)}\n"
    exploration_text += f"Total search points: {len(all_candidates)}\n"
    exploration_text += f"Successful selections: {sum(is_selected)}"

    ax.text(0.02, 0.98, exploration_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.tight_layout()
    os.makedirs('extension_plots', exist_ok=True)
    plt.savefig('extension_plots/search_space_exploration.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_prompt_component_analysis(best_values):
    """Prompt component contribution analysis."""
    if 'ape_tracking' not in best_values:
        print("❌ No tracking data available for prompt component analysis")
        return

    tracking_data = best_values['ape_tracking']
    all_candidates = tracking_data['all_candidates']

    if not all_candidates:
        print("❌ No candidate data available")
        return

    template_performance = {}
    healthy_term_performance = {}
    disease_term_performance = {}

    for candidate in all_candidates:
        template = candidate['template'] if candidate['template'] else '(empty)'
        healthy = candidate['descriptions']['No retinopathy']
        disease = candidate['descriptions']['Retinopathy']
        f1 = candidate['f1']

        template_performance.setdefault(template, []).append(f1)
        healthy_term_performance.setdefault(healthy, []).append(f1)
        disease_term_performance.setdefault(disease, []).append(f1)

    template_avg = {k: np.mean(v) for k, v in template_performance.items()}
    healthy_avg = {k: np.mean(v) for k, v in healthy_term_performance.items()}
    disease_avg = {k: np.mean(v) for k, v in disease_term_performance.items()}

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    def _make_bar(ax, avg_dict, perf_dict, color, title):
        items = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)
        labels, scores = zip(*items)
        counts = [len(perf_dict[l]) for l in labels]
        bars = ax.barh(range(len(labels)), scores, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        for i, (score, count) in enumerate(zip(scores, counts)):
            ax.text(score + 0.005, i, f'n={count}', va='center', fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l[:30] + '...' if len(l) > 30 else l for l in labels], fontsize=10)
        ax.set_xlabel('Average F1-score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        if bars:
            bars[0].set_color('#FFD700')
            bars[0].set_alpha(1.0)

    _make_bar(ax1, template_avg, template_performance, 'lightblue', 'Template component performance')
    _make_bar(ax2, healthy_avg, healthy_term_performance, 'lightgreen', 'Healthy term component performance')
    _make_bar(ax3, disease_avg, disease_term_performance, 'lightcoral', 'Disease term component performance')

    stats_text = (f"Component diversity:\nTemplates: {len(template_avg)}\n"
                  f"Healthy terms: {len(healthy_avg)}\nDisease terms: {len(disease_avg)}")
    fig.text(0.02, 0.98, stats_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    os.makedirs('extension_plots', exist_ok=True)
    plt.savefig('extension_plots/prompt_component_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def create_final_extension_plots(best_values):
    """Create all final extension plots for journal submission."""
    _setup_plot_style()
    print("🎨 Creating final extension plots for journal submission...")

    plot_comprehensive_performance_analysis(best_values)
    plot_computational_efficiency(best_values)
    plot_learning_progression(best_values)
    plot_candidate_quality_distribution(best_values)
    plot_search_space_exploration(best_values)
    plot_prompt_component_analysis(best_values)

    print("✅ All final extension plots created!")
    print(f"📁 Plots saved in: ./extension_plots/")
    print("\nPlot summary:")
    print("  1. performance_analysis.pdf - Learning vs transfer across tasks")
    print("  2. computational_efficiency.pdf - Efficiency comparison")
    print("  3. learning_progression.pdf - Start vs final stages evolution")
    print("  4. candidate_quality_distribution.pdf - Solution space exploration")
    print("  5. search_space_exploration.pdf - Novel prompt landscape navigation")
    print("  6. prompt_component_analysis.pdf - Breakdown of prompt components")
