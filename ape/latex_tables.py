import pandas as pd
import numpy as np


def generate_latex_table(df: pd.DataFrame, save_path: str = None) -> str:
    """Generate LaTeX tables comparing Original, TADILER, and APE methods"""

    df = df.copy()
    df['Method_Type'] = df['Method'].apply(lambda x:
        'APE' if '-APE' in x
        else 'TADILER' if '-TADILER' in x
        else 'Original'
    )

    architectures = ['Attention', 'Residual', 'MLP']
    knn_values = [15, 20, 25, 30, 50]

    strategy_groups = [
        (['Naive', 'GEM'], 'naive_gem'),
        (['LwF', 'EWC'], 'lwf_ewc')
    ]

    all_tables = []

    for group_idx, (strategies, group_name) in enumerate(strategy_groups):
        latex_lines = []

        table_num = group_idx + 1
        strategy_names = " and ".join(strategies)

        latex_lines.extend([
            f"\\begin{{table*}}",
            "\\scriptsize",
            f"  {{\\caption{{Comparison of Mean AMCA scores and forgetting (parentheses) for {strategy_names} strategies across different base models. Boldface",
            "indicates superior performance among \\textit{Original}, \\textit{TADILER}, and \\textit{APE} methods based on Mean AMCA. Averages are",
            "computed across different NN values for each base model.}",
            f"  \\label{{tab:amca_comparison_{group_name}}}",
            "}",
            "  {\\begin{tabular}{llcccccc}",
            "  ",
            "  \\toprule"
        ])

        strategy_headers = []
        cmidrule_parts = []
        method_headers = []

        col_start = 3
        for i, strategy in enumerate(strategies):
            strategy_headers.append(f"\\multicolumn{{3}}{{c}}{{\\bfseries {strategy}}}")
            cmidrule_parts.append(f"\\cmidrule(lr){{{col_start}-{col_start+2}}}")
            method_headers.extend([f"\\bfseries Original", f"\\bfseries TADILER", f"\\bfseries APE"])
            col_start += 3

        latex_lines.extend([
            f"  \\bfseries Model & \\bfseries NN & {' & '.join(strategy_headers)} \\\\",
            f"  {' '.join(cmidrule_parts)}",
            f" &  \\bfseries  & {' & '.join(method_headers)} \\\\",
            "  \\midrule"
        ])

        def format_value_with_forgetting(amca, forgetting):
            return f"{amca:.3f}({abs(forgetting):.1f})"

        def get_method_data(subset_df, method_type):
            method_data = subset_df[subset_df['Method_Type'] == method_type]
            if method_data.empty:
                return None, None
            amca = method_data['Mean AMCA'].iloc[0]
            forgetting = method_data['Forgetting (0-10)'].iloc[0]
            return amca, forgetting

        def bold_max_values(values_list, method_names=None):
            if method_names is None:
                method_names = ['Original', 'TADILER', 'APE']
            if not values_list or all(v is None for v in values_list):
                return values_list

            amca_values = []
            forgetting_values = []

            for val in values_list:
                if val is None:
                    amca_values.append(-1)
                    forgetting_values.append(999)
                else:
                    parts = val.split('(')
                    amca_val = float(parts[0])
                    forgetting_val = float(parts[1].rstrip(')'))
                    amca_values.append(amca_val)
                    forgetting_values.append(forgetting_val)

            max_amca = max(amca_values)
            max_amca_indices = [i for i, val in enumerate(amca_values) if val == max_amca]

            if len(max_amca_indices) == 1:
                best_idx = max_amca_indices[0]
            else:
                min_forgetting_among_max = min(forgetting_values[i] for i in max_amca_indices)
                forgetting_tie_indices = [i for i in max_amca_indices
                                        if forgetting_values[i] == min_forgetting_among_max]

                if len(forgetting_tie_indices) == 1:
                    best_idx = forgetting_tie_indices[0]
                else:
                    ape_indices = [i for i in forgetting_tie_indices
                                 if method_names[i] == 'APE']
                    if ape_indices:
                        best_idx = ape_indices[0]
                    else:
                        best_idx = forgetting_tie_indices[0]

            result = []
            for i, val in enumerate(values_list):
                if val is None:
                    result.append("--")
                elif i == best_idx:
                    result.append(f"\\textbf{{{val}}}")
                else:
                    result.append(val)

            return result

        for arch_idx, architecture in enumerate(architectures):
            arch_df = df[df['Architecture'] == architecture]

            first_row_prefix = f"\\multirow{{7}}{{*}}{{\\textbf{{{architecture}}}}}"

            for knn_idx, knn in enumerate(knn_values):
                knn_df = arch_df[arch_df['KNN'] == knn]

                if knn_idx == 0:
                    line = f"{first_row_prefix} & {knn} "
                else:
                    line = f" & {knn} "

                for strategy in strategies:
                    strategy_df = knn_df[knn_df['Strategy'] == strategy]

                    orig_amca, orig_forg = get_method_data(strategy_df, 'Original')
                    tadiler_amca, tadiler_forg = get_method_data(strategy_df, 'TADILER')
                    ape_amca, ape_forg = get_method_data(strategy_df, 'APE')

                    orig_val = format_value_with_forgetting(orig_amca, orig_forg) if orig_amca is not None else None
                    tadiler_val = format_value_with_forgetting(tadiler_amca, tadiler_forg) if tadiler_amca is not None else None
                    ape_val = format_value_with_forgetting(ape_amca, ape_forg) if ape_amca is not None else None

                    bold_vals = bold_max_values([orig_val, tadiler_val, ape_val])

                    line += f"& {bold_vals[0]} & {bold_vals[1]} & {bold_vals[2]} "

                line += "\\\\"
                latex_lines.append(line)

            latex_lines.append(f"  \\cmidrule(lr){{2-{2 + len(strategies) * 3}}}")
            avg_line = " & \\textbf{Avg} "

            for strategy in strategies:
                strategy_df = arch_df[arch_df['Strategy'] == strategy]

                orig_avg_amca = strategy_df[strategy_df['Method_Type'] == 'Original']['Mean AMCA'].mean()
                orig_avg_forg = strategy_df[strategy_df['Method_Type'] == 'Original']['Forgetting (0-10)'].mean()

                tadiler_avg_amca = strategy_df[strategy_df['Method_Type'] == 'TADILER']['Mean AMCA'].mean()
                tadiler_avg_forg = strategy_df[strategy_df['Method_Type'] == 'TADILER']['Forgetting (0-10)'].mean()

                ape_avg_amca = strategy_df[strategy_df['Method_Type'] == 'APE']['Mean AMCA'].mean()
                ape_avg_forg = strategy_df[strategy_df['Method_Type'] == 'APE']['Forgetting (0-10)'].mean()

                orig_avg = format_value_with_forgetting(orig_avg_amca, orig_avg_forg) if not np.isnan(orig_avg_amca) else None
                tadiler_avg = format_value_with_forgetting(tadiler_avg_amca, tadiler_avg_forg) if not np.isnan(tadiler_avg_amca) else None
                ape_avg = format_value_with_forgetting(ape_avg_amca, ape_avg_forg) if not np.isnan(ape_avg_amca) else None

                bold_avgs = bold_max_values([orig_avg, tadiler_avg, ape_avg])

                avg_line += f"& {bold_avgs[0]} & {bold_avgs[1]} & {bold_avgs[2]} "

            avg_line += "\\\\"
            latex_lines.append(avg_line)

            if arch_idx < len(architectures) - 1:
                latex_lines.append("  \\midrule")

        latex_lines.extend([
            "  \\midrule",
            "  \\bottomrule",
            "  \\end{tabular}}",
            "\\end{table*}",
            ""
        ])

        all_tables.append('\n'.join(latex_lines))

    final_latex = '\n'.join(all_tables)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(final_latex)
        print(f"LaTeX tables saved to {save_path}")

    return final_latex


def generate_separate_tables(df: pd.DataFrame, save_path_1: str = None, save_path_2: str = None):
    """Generate two separate LaTeX files for each table"""

    full_latex = generate_latex_table(df)

    tables = full_latex.split('\\end{table*}\n\n\\begin{table*}')

    if len(tables) == 2:
        table1 = tables[0] + '\\end{table*}'
        table2 = '\\begin{table*}' + tables[1]

        if save_path_1:
            with open(save_path_1, 'w') as f:
                f.write(table1)
            print(f"Table 1 (Naive & GEM) saved to {save_path_1}")

        if save_path_2:
            with open(save_path_2, 'w') as f:
                f.write(table2)
            print(f"Table 2 (LwF & EWC) saved to {save_path_2}")

        return table1, table2

    return full_latex, ""


def print_latex_table(df: pd.DataFrame):
    """Print both LaTeX tables to console"""
    latex_tables = generate_latex_table(df)
    print(latex_tables)


# ---------------------------------------------------------------------------
# APE-specific LaTeX tables (ablation + random baseline)
# ---------------------------------------------------------------------------

def generate_llm_ablation_latex(
    ablation_results: dict,
    random_results: dict | None = None,
    medgemma_scores: tuple = (0.9229, 0.917, 0.918),
    tadiler_scores: tuple = (0.8921, 0.8899, 0.8909),
    save_path: str = None,
) -> str:
    """Generate the LLM backbone ablation LaTeX table (Table llm_ablation).

    Args:
        ablation_results: Return value of ``run_llm_ablation()``.
            Keys are model labels; each value contains f1_task0, f1_task1, f1_task2.
        random_results: Dict with keys 'f1_task0', 'f1_task1', 'f1_task2'
            from the random baseline run (or None to leave as dashes).
        medgemma_scores: (f1_t0, f1_t1, f1_t2) for Med-Gemma 4B (reference row).
        tadiler_scores: (f1_t0, f1_t1, f1_t2) for TADILER exhaustive search.
        save_path: If given, write the table to this file.

    Returns:
        LaTeX string ready to paste into the paper.
    """
    n_tasks = max(
        max(int(k.replace("f1_task", "")) for k in v if k.startswith("f1_task"))
        for v in ablation_results.values()
    ) + 1

    # Build all rows: (display_name, [f1_t0, f1_t1, ...], is_reference)
    rows = []
    rows.append(("Med-Gemma 4B (main method)", list(medgemma_scores[:n_tasks]), True))
    for label, data in ablation_results.items():
        scores = [data.get(f"f1_task{i}", None) for i in range(n_tasks)]
        rows.append((label, scores, False))
    rows.append(("TADILER exhaustive search", list(tadiler_scores[:n_tasks]), True))
    if random_results is not None:
        rand_scores = [random_results.get(f"f1_task{i}", None) for i in range(n_tasks)]
        rows.append(("Random selection (baseline)", rand_scores, True))
    else:
        rows.append(("Random selection (baseline)", [None] * n_tasks, True))

    # Find best score per task column (among the middle rows = ablation models)
    ablation_f1s = [
        [data.get(f"f1_task{i}") for i in range(n_tasks)]
        for data in ablation_results.values()
    ]

    def _fmt(val):
        return f"{val:.4f}" if val is not None else "--"

    def _bold_best_in_col(rows_data, col_idx):
        """Return the row index of the best value in a column (ablation rows only)."""
        ablation_indices = [i for i, (_, _, ref) in enumerate(rows_data) if not ref]
        best_val = max(
            (rows_data[i][1][col_idx] for i in ablation_indices if rows_data[i][1][col_idx] is not None),
            default=None,
        )
        return best_val

    task_headers = " & ".join(f"\\bfseries Task~{i}" for i in range(n_tasks))
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{LLM backbone ablation. F1-score on Task~0 (evolution) and Tasks~1 "
        "and~2 (frozen prompt transfer). Med-Gemma~4B and TADILER exhaustive search "
        "are reference rows. Random selection uses the same 50-evaluation budget as APE.}",
        "\\label{tab:llm_ablation}",
        f"\\begin{{tabular}}{{l{'c' * n_tasks}}}",
        "\\toprule",
        f"\\bfseries Model & {task_headers} \\\\",
        "\\midrule",
    ]

    best_per_task = [
        _bold_best_in_col(rows, col) for col in range(n_tasks)
    ]

    prev_was_reference = None
    for i, (name, scores, is_ref) in enumerate(rows):
        # Insert midrule to separate reference rows from ablation models
        if prev_was_reference is not None and is_ref != prev_was_reference:
            lines.append("\\midrule")
        prev_was_reference = is_ref

        cells = []
        for col, val in enumerate(scores):
            formatted = _fmt(val)
            if (not is_ref) and (val is not None) and (best_per_task[col] is not None) and abs(val - best_per_task[col]) < 1e-9:
                formatted = f"\\textbf{{{formatted}}}"
            cells.append(formatted)

        row_str = name + " & " + " & ".join(cells) + " \\\\"
        lines.append(row_str)

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    latex = "\n".join(lines)
    if save_path:
        with open(save_path, "w") as f:
            f.write(latex)
        print(f"LLM ablation table saved to {save_path}")
    return latex


def generate_random_baseline_latex(
    random_f1s: dict,
    medgemma_scores: tuple = (0.9229, 0.917, 0.918),
    save_path: str = None,
) -> str:
    """Generate a compact LaTeX snippet comparing APE vs. random baseline per task.

    Args:
        random_f1s: Dict with keys 'f1_task0', 'f1_task1', 'f1_task2'.
        medgemma_scores: APE (Med-Gemma) F1 scores per task.
        save_path: Optional path to save the snippet.

    Returns:
        LaTeX tabular string.
    """
    n_tasks = len(medgemma_scores)

    lines = [
        "% Random baseline comparison snippet — paste into Section 5.1",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Budget-matched random prompt selection baseline vs.\\ APE (Med-Gemma 4B). "
        "Both methods use 50 evaluations. Task~0 is the evolution task; "
        "Tasks~1--2 use the frozen best prompt.}",
        "\\label{tab:random_baseline}",
        f"\\begin{{tabular}}{{l{'c' * n_tasks}}}",
        "\\toprule",
        "\\bfseries Method & " + " & ".join(f"\\bfseries Task~{i}" for i in range(n_tasks)) + " \\\\",
        "\\midrule",
    ]

    ape_cells = " & ".join(f"{medgemma_scores[i]:.4f}" for i in range(n_tasks))
    rand_cells = " & ".join(
        f"{random_f1s.get(f'f1_task{i}', 0.0):.4f}" if random_f1s.get(f"f1_task{i}") is not None else "--"
        for i in range(n_tasks)
    )
    delta_cells = " & ".join(
        f"{medgemma_scores[i] - random_f1s.get(f'f1_task{i}', medgemma_scores[i]):+.4f}"
        if random_f1s.get(f"f1_task{i}") is not None else "--"
        for i in range(n_tasks)
    )

    lines += [
        f"APE (Med-Gemma 4B) & {ape_cells} \\\\",
        f"Random selection   & {rand_cells} \\\\",
        "\\midrule",
        f"$\\Delta$ (APE $-$ Random) & {delta_cells} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    latex = "\n".join(lines)
    if save_path:
        with open(save_path, "w") as f:
            f.write(latex)
        print(f"Random baseline table saved to {save_path}")
    return latex
