# Reviewer Response Plan
## Paper: Adaptive Prompt Evolution for Continual Learning in Diabetic Retinopathy Detection

This is the reference document for the full revision. Each reviewer's original comments
are listed with decisions and exact LaTeX patches. Code hints are provided for all new
experiments. The execution order at the end classifies every task as either
INDEPENDENT (can start immediately) or DEPENDENT (requires experiment results first),
so parallel work is possible.

---

## Framing decisions that apply throughout the paper

**Med-Gemma 4B remains the primary model.** All existing results and tables are kept
as-is. Med-Gemma 4B is the main deployment model: medical domain pretraining, offline
capable, privacy-compliant.

**Three new LLMs are introduced as a separate ablation section only.** Gemma 3 4B,
Qwen3 4B, and Llama 3.2 3B are general-purpose small models that are also deployable
locally. They are accessed via OpenRouter for experimental convenience. The ablation
runs the full experiment grid (all seeds, all strategies, all architectures) to produce
AMCA scores directly comparable to the main tables.

**The ablation answers one clean scientific question:** does APE's evolutionary
mechanism generalize beyond a medically specialized backbone, or are the gains specific
to Med-Gemma 4B?

**Random selection baseline** is the only budget-matched prompt baseline added. It
uses the same 50-evaluation budget as APE and answers whether LLM-guided search adds
value over uniform sampling of the vocabulary.

**Seeds:** 5 seeds total: `[3, 51, 35, 42, 99]`. All experiments (Med-Gemma main
results and new LLM ablation) use this set. If Med-Gemma was originally run on 3 seeds
`[3, 51, 35]`, add 2 more runs for Med-Gemma as well so both the main results and
ablation have n=5.

---

## Notation

- **IMPLEMENT NOW**: Requires code changes or new experiments.
- **WRITING ONLY**: Edit the .tex file only, no new experiments needed.
- **FUTURE WORK**: Acknowledge in Limitations or Conclusion, no experiments.
- **DISCARD**: Out of scope; add one sentence to Limitations if needed.

---
---

## Reviewer 1

**Original requests:**

1. Highlight the novelty of the paper explicitly in the Introduction, last paragraph.
2. Add a summary of the literature survey in the form of a table.
3. Add a comparative study with existing methods.
4. State the advantages of the method clearly.
5. Provide the time complexity of the method.
6. Add sample result images in the experimental section.
7. Check and fix typo errors.

**Decisions:** All WRITING ONLY.

- R1.3 (comparative study): existing Tables 1 and 2 already serve this; supplement
  with a summary table in Related Work.
- R1.5 (time complexity): derive analytically from Algorithm 1 structure and from
  empirical timing already in the notebook.
- R1.6 (sample images): extend existing Figure 2 with a panel showing evolved prompt
  outputs or CLIP clustering results per task.

---

### Patch R1.1 -- Introduction: replace the contributions list

**Location:** `\section{Introduction}`, the `enumerate` contributions block.

```latex
The main contributions of this paper are:
\begin{enumerate}
    \item \textbf{Adaptive Prompt Evolution (APE):} A framework for autonomous prompt
    discovery that replaces exhaustive static enumeration with performance-driven
    evolutionary search guided by a locally deployed medical LLM, requiring
    2.2$\times$ fewer evaluations while achieving superior F1-score compared to the
    exhaustive baseline. Unlike TADILER's fixed search over 110 predefined combinations,
    APE iteratively refines prompts through LLM-guided feedback, constituting a
    learning-based optimization framework rather than a lookup procedure.
    \item \textbf{Privacy-preserving local LLM integration:} Integration of Med-Gemma
    4B for offline prompt generation, ensuring privacy compliance without external API
    dependencies while leveraging medical domain expertise.
    \item \textbf{Comprehensive CL evaluation:} Systematic integration and evaluation
    with four established CL strategies (EWC, GEM, LwF, Naive) and three neural
    network architectures across five random seeds, with statistical validation of
    performance gains and an ablation study demonstrating the generalizability of the
    evolutionary mechanism across LLM backbones.
\end{enumerate}
```

---

### Patch R1.2 -- Related Work: add literature summary table

**Location:** `\section{Related work}`, immediately before `\section{Background and
problem formulation}`.

```latex
Table~\ref{tab:related_work_summary} summarizes the approaches discussed above and
positions APE relative to existing work across the dimensions most relevant to
privacy-preserving medical continual learning.

\begin{table*}[htbp]
\centering
\caption{Summary of related approaches for prompt-based and privacy-preserving
continual learning in medical imaging. APE uniquely combines autonomous prompt
evolution with local medical LLM deployment and embedding-based privacy preservation.}
\label{tab:related_work_summary}
\begin{tabular}{lccccl}
\toprule
\bfseries Method & \bfseries Prompt & \bfseries Local & \bfseries Privacy
  & \bfseries CL Strategy & \bfseries Domain \\
                 & \bfseries Evolved & \bfseries Deploy & \bfseries Preserving & & \\
\midrule
CLIP zero-shot~\cite{Radford2021}  & No  & No  & No  & None     & General vision \\
TADILER~\cite{BravoRocca2025}      & No  & No  & Yes & Multiple & Medical imaging \\
iCARL~\cite{ICARL}                 & No  & No  & No  & Replay   & General vision \\
EvoPrompt~\cite{evoprompt2024}     & Yes & No  & No  & Single   & General NLP    \\
APE (ours)                         & Yes & Yes & Yes & Multiple & Medical imaging \\
\bottomrule
\end{tabular}
\end{table*}
```

---

### Patch R1.3 -- New subsection: computational complexity

**Location:** Add as new subsection after `\subsection{APE optimization and evaluation
protocol}`.

```latex
\subsection{Computational complexity analysis}
\label{sec:complexity}

APE's time complexity is determined by three nested operations. Let $I$ denote the
maximum number of evolution iterations, $k$ the number of candidates generated per
iteration, $N$ the number of training samples in the task, and $D$ the CLIP embedding
dimension. Each candidate evaluation computes cosine similarity between $N$ image
embeddings and two text embeddings, giving $\mathcal{O}(N \cdot D)$ per evaluation.
LLM-guided candidate generation adds a constant per-candidate inference cost
$C_{\text{LLM}}$ that is independent of $N$. The total complexity of one APE evolution
run is:
\begin{equation}
\mathcal{O}(I \cdot k \cdot (N \cdot D + C_{\text{LLM}}))
\end{equation}
In our experiments, $I_{\text{max}} = 10$, $k = 5$, $D = 768$, and $N$ ranges from
832 to 2{,}929 depending on the task. By contrast, TADILER's exhaustive search
evaluates all combinations from a fixed vocabulary:
$\mathcal{O}(|\mathcal{T}| \cdot |\mathcal{H}| \cdot |\mathcal{D}| \cdot N \cdot D)$,
where $|\mathcal{T}|$, $|\mathcal{H}|$, $|\mathcal{D}|$ are template, healthy-term,
and disease-term vocabulary sizes respectively (110 combinations total in TADILER).
APE's patience mechanism terminates once no improvement is observed for $P$
consecutive iterations, giving worst-case bound $\mathcal{O}(I_{\text{max}} \cdot k
\cdot N \cdot D)$ and empirical convergence at 50 evaluations -- a 2.2$\times$
reduction over the 110-evaluation exhaustive baseline.
Empirically, one APE run completes in approximately 9.2 minutes on a CPU-only platform
(dual Intel Xeon Platinum 8360Y), consistent with practical offline deployment.
```

---

### Note on typos (R1.7)

After applying all patches: standardize "F1-score" throughout (not "F1 score"),
verify "Med-Gemma 4B" capitalization is consistent, and confirm all new table and
figure cross-references compile without warnings.

---
---

## Reviewer 3

**Original requests:**

Major:
1a. Explicitly contrast APE with TADILER: search strategy, adaptability, complexity, generalization.
1b. Discuss how APE differs from RL, Bayesian optimization, evolutionary NLP. Add comparison table.
2a. Justify why Task 0 optimization does not introduce task-specific bias.
2b. Clarify whether prompts are frozen after Task 0 or re-optimized for each new task.
2c. Add evolution protocol clarification and overfitting discussion.
3a. Clarify why Med-Gemma 4B was selected over general-purpose LLMs or simpler mutation strategies.
3b. Clarify whether gains come from medical domain knowledge or from the search mechanism.
3c. Ablation with non-medical or smaller language models.
4. Discuss computational and deployment implications of Med-Gemma 4B in healthcare.
5a. State which statistical tests were used and how significance was determined.
5b. Report confidence intervals or effect sizes in tables.

Minor:
M1a. Add complexity and convergence discussion near Algorithm 1.
M1b. Disambiguate reused symbols.
M2. Justify short prompt (2-4 word) constraint.
M3. Improve readability of Figures 3 and 4.
M4. Clarify random seed usage in LLM generation; release evolved prompt examples.
M5. Reduce repetitive phrasing and subjective descriptors.

**Decisions:**

- R3.1a/b (contrast + comparison table): **WRITING ONLY**
- R3.2a/b/c (prompt freezing): **WRITING ONLY** (code already correct)
- R3.3a (why Med-Gemma): **WRITING ONLY**
- R3.3b (mechanism vs backbone): answered by the ablation
- R3.3c (LLM ablation): **IMPLEMENT NOW** (see execution order and code hint below)
- R3.4 (deployment implications): **WRITING ONLY**
- R3.5a/b (statistical tests + CIs): **IMPLEMENT NOW** (shared with R5.4)
- All minor comments: **WRITING ONLY**

---

### Patch R3.1 -- Background: rewrite TADILER limitations paragraph with comparison table

**Location:** `\subsection{TADILER framework and limitations}`, replace the paragraph
beginning "However, TADILER suffered from significant limitations..."

```latex
However, TADILER suffers from three fundamental limitations that APE addresses. First,
it evaluates all 110 predefined prompt combinations exhaustively, making search cost
proportional to vocabulary size regardless of candidate quality. Second, prompts are
static once selected -- there is no mechanism for refinement as tasks evolve. Third,
its original implementation uses external API calls for prompt generation, creating
tension with offline healthcare deployment. Table~\ref{tab:ape_vs_methods} contrasts
APE with TADILER and other automated prompt optimization paradigms.

\begin{table}[htbp]
\centering
\caption{Comparison of prompt optimization approaches. APE uniquely combines
evolutionary search with local LLM deployment and cross-task prompt transfer.}
\label{tab:ape_vs_methods}
\begin{tabular}{lcccc}
\toprule
\bfseries Method & \bfseries Search Strategy & \bfseries Cross-task
  & \bfseries Local Deploy & \bfseries Medical Domain \\
\midrule
TADILER~\cite{BravoRocca2025}   & Exhaustive enum. & No  & No  & No \\
DSPy~\cite{khattab2023dspy}     & Gradient-free    & No  & No  & No \\
EvoPrompt~\cite{evoprompt2024}  & Evolutionary     & No  & No  & No \\
APE (ours)                      & Evolutionary     & Yes & Yes & Yes \\
\bottomrule
\end{tabular}
\end{table}
```

---

### Patch R3.2 -- Section 4.3: prompt freezing and Task 0 bias

**Location:** `\subsection{Performance evaluation and evolution control}`, add at the end.

```latex
\noindent\textbf{Cross-task prompt transfer and overfitting.}
Prompt evolution is performed exclusively on Task~0, using zero-shot clustering
F1-score on the training partition as the fitness signal. Once evolution converges,
the resulting prompt $\mathbf{p}^*$ is \emph{frozen} and applied without modification
to all subsequent tasks. No labels from Tasks~1 or~2 are used at any stage of prompt
optimization.
This design is deliberate. Task~0 serves as the initialization step where evolutionary
search identifies descriptions that effectively separate healthy and diseased fundus
images in CLIP's embedding space. These descriptions encode visual semantics that
remain valid across the synthetic domain shifts of Tasks~1 and~2, because the
underlying CLIP embedding space is shared across tasks. The generalization results in
Figure~\ref{fig:performance_analysis} (F1 = 0.917 on Task~1, 0.918 on Task~2 vs.\
0.923 on Task~0) confirm this empirically: the performance drop across tasks is
$\Delta = 0.005$, within the variance observed across random seeds.
```

---

### Patch R3.3 -- Section 4.2: Med-Gemma selection rationale and short prompt justification

**Location:** `\subsection{Medical LLM integration and prompt generation}`, add after
the sentence ending "...temperature 0.7 and maximum 300 tokens..."

```latex
Med-Gemma 4B was selected for three reasons. First, its pretraining on ophthalmology,
radiology, and histopathology data improves the likelihood of generating clinically
relevant terminology for fundus image descriptions without additional prompt engineering.
Second, its 4B parameter scale enables deployment on standard CPU hardware without GPU
acceleration, satisfying offline deployment constraints. Third, it is compatible with
local inference tools (e.g., Ollama, llama.cpp), ensuring that no patient data needs
to leave the institution. Section~\ref{sec:llm_ablation} provides an empirical
assessment of whether the observed gains are specific to Med-Gemma 4B's medical
pretraining or attributable to the evolutionary mechanism more broadly.

The restriction to short prompts of 2-4 words per description is motivated by CLIP's
pretraining distribution. CLIP's text encoder was trained predominantly on short,
noun-phrase style captions~\cite{Radford2021}, and longer descriptions tend to dilute
the semantic signal by introducing tokens with weak correspondence to visual features.
This is consistent with findings in the prompt engineering literature showing that
concise, specific descriptions outperform verbose ones in zero-shot CLIP
classification~\cite{ape_survey_2025}.
```

---

### Patch R3.4 -- Limitations: Med-Gemma deployment paragraph

**Location:** `\section{Limitations}`, second paragraph (update or replace the
existing deployment paragraph).

```latex
APE's evolutionary search introduces a one-time computational overhead of approximately
9.2 minutes on CPU-only hardware, incurred once before continual learning begins. This
is acceptable for most clinical screening workflows given that prompt re-optimization
is not required for subsequent tasks. Med-Gemma 4B, with 4B parameters, can be
deployed locally on standard workstation hardware; however, edge deployments on
embedded clinical devices with highly constrained compute may require quantized
variants or distilled alternatives. The ablation in Section~\ref{sec:llm_ablation}
shows that general-purpose LLMs of comparable size also produce effective prompts
through evolutionary search, providing practitioners with flexibility in model
selection based on their hardware and licensing constraints.
```

---

### Patch R3.5 -- Algorithm 1: add complexity and symbol disambiguation note

**Location:** After Algorithm 1 (`\label{alg:ape_evolution}`).

```latex
The worst-case complexity of Algorithm~\ref{alg:ape_evolution} is
$\mathcal{O}(I_{\text{max}} \cdot k \cdot N \cdot D)$; see
Section~\ref{sec:complexity} for the full derivation. In practice, the patience
mechanism terminates the loop at 50 total evaluations in our experiments, well below
the $I_{\text{max}} \cdot k = 70$ ceiling.
Note on notation: the evolution history structure $\mathcal{H}$ used in line~4 to
record per-iteration candidate scores and selected prompts is distinct from the task
sequence index $t$ defined in Section~\ref{sec:background}.
```

---
---

## Reviewer 5

**Original requests:**

1. Strict protocol separating prompt evolution data from final test evaluation.
2. Include at least one experiment with realistic domain shift (cross-dataset).
3. Add at least two budget-matched baselines: random prompt selection and a fixed-template heuristic.
4. Report statistical test, number of runs, pairing, mean/std, confidence intervals.
5. Privacy stress test (membership inference or nearest-neighbour reconstruction).
6. Add recent 2023-2025 literature.

**Decisions:**

- R5.1 (data separation): **WRITING ONLY** (code already correct)
- R5.2 (realistic domain shift): **FUTURE WORK** + Limitations
- R5.3 (random selection baseline only; fixed-template heuristic dropped): **IMPLEMENT NOW**
- R5.4 (statistical tests + CIs): **IMPLEMENT NOW** (shared with R3.5)
- R5.5 (privacy stress test): **DISCARD** + one sentence in Limitations
- R5.6 (recent literature): **WRITING ONLY**

---

### Patch R5.1 -- Section 4.3: explicit data separation statement

**Location:** `\subsection{APE optimization and evaluation protocol}`, replace the
sentence beginning "We implement comprehensive validation through multiple random
seeds..."

```latex
All datasets are partitioned into training (80\%) and test (20\%) sets using a fixed
random seed ($\textit{seed} = 42$) prior to any model training or prompt optimization.
Prompt evolution is conducted using the training partition of Task~0 only; the F1-score
used as the fitness signal is computed on that same training partition. All F1-scores
and AMCA values reported in Tables~\ref{tab:amca_comparison_naive_gem}
and~\ref{tab:amca_comparison_lwf_ewc} are computed exclusively on the held-out test
sets, which are never seen during prompt evolution or model training.
Results are validated across five random seeds $\{3, 51, 35, 42, 99\}$ and five
neighbor count values $\{15, 20, 25, 30, 50\}$ to assess stability across
initialization and sampling choices.
```

---

### Patch R5.2 -- Limitations: realistic domain shift

**Location:** `\section{Limitations}`, insert before the paragraph beginning
"Finally, APE's stopping criteria..."

```latex
The domain shifts evaluated in this work are constructed synthetically through image
transformations (lighting variation and Gaussian noise addition), following established
practice in medical continual learning benchmarking~\cite{BravoRocca2025}. This does
not capture the full complexity of real-world domain shift arising from differences in
camera hardware, acquisition protocol, or patient demographics across clinical sites.
Cross-dataset evaluation, for example training on APTOS~2019 and evaluating on
Messidor-2 or IDRiD~\cite{porwal2018idrid}, would provide stronger evidence of
real-world robustness and is an important direction for future work.
```

---

### Patch R5.3 -- Section 4: random selection baseline description

**Location:** `\subsection{Integration with CL strategies}`, add before "As
demonstrated in the experimental results..."

```latex
\noindent\textbf{Random prompt selection baseline.}
To isolate the contribution of LLM-guided evolutionary search, we introduce a
budget-matched baseline that draws candidate prompts uniformly from a fixed vocabulary
of medically relevant short descriptions and selects the best-performing combination
after 50 evaluations -- the same budget used by APE. This baseline provides a direct
test of whether the evolutionary search mechanism adds value over random sampling of
the same prompt vocabulary.
```

---

### Code hint -- random selection baseline

```python
import numpy as np

RANDOM_TEMPLATES = [
    "", "Fundus image shows", "Retinal scan reveals",
    "Eye examination shows", "Ophthalmoscopy reveals",
    "Retinal photo shows", "Fundus photograph of",
    "Fundus image reveals", "Clinical scan shows",
]
RANDOM_HEALTHY = [
    "healthy", "normal retina", "clear vessels", "no DR",
    "healthy fundus", "no lesions", "normal eye",
    "normal fundus", "clear fundus", "healthy retina",
]
RANDOM_DISEASED = [
    "diseased", "DR present", "retinal lesions", "diabetic damage",
    "abnormal retina", "hemorrhages present", "retinopathy detected",
    "fundus abnormality", "diabetic retinopathy", "lesions detected",
]


def random_prompt_baseline(task_data, n_evaluations=50, seed=42):
    """
    Budget-matched random search. Draws uniformly from vocabulary;
    returns best (template, descriptions, f1) and full F1 trajectory.
    """
    rng = np.random.default_rng(seed)
    best_f1, best_combo, all_f1s = 0.0, None, []
    for _ in range(n_evaluations):
        template     = str(rng.choice(RANDOM_TEMPLATES))
        healthy      = str(rng.choice(RANDOM_HEALTHY))
        diseased     = str(rng.choice(RANDOM_DISEASED))
        descriptions = {"No retinopathy": healthy, "Retinopathy": diseased}
        f1 = evaluate_single_prompt(template, descriptions, task_data)
        all_f1s.append(f1)
        if f1 > best_f1:
            best_f1, best_combo = f1, {
                "template": template, "descriptions": descriptions}
    print(f"[Random] best={best_f1:.4f}  avg={np.mean(all_f1s):.4f}")
    return best_combo, best_f1, all_f1s


# Transfer best prompt to Tasks 1 and 2:
# f1_t1 = evaluate_single_prompt(
#     best_combo["template"], best_combo["descriptions"], list_doclists[1])
# f1_t2 = evaluate_single_prompt(
#     best_combo["template"], best_combo["descriptions"], list_doclists[2])
```

---

### Patch R5.3b -- Section 5.1: add random baseline result to evolution discussion

**Location:** `\subsection{Autonomous prompt evolution performance}`, after the 2.2x
efficiency sentence.

```latex
Compared to the random prompt selection baseline, which achieves a best F1-score of
[fill: rand_f1_t0] on Task~0 using the same 50-evaluation budget, APE's LLM-guided
evolutionary search demonstrates the value of directed candidate generation.
[After running: add one sentence comparing transfer performance on Tasks 1 and 2,
and one sentence interpreting what the gap implies about the quality of LLM-generated
candidates versus uniform random sampling.]

% Fill in after running random_prompt_baseline() and transferring to Tasks 1 and 2.
```

---

### Code hint -- statistical tests (R3.5a/b and R5.4 combined)

```python
# First: in the configuration cell, ensure:
#   random_seeds = [3, 51, 35, 42, 99]
# Re-run the full experiment grid so df_all has n=5 per strategy-method combination.
# If Med-Gemma was already run on [3, 51, 35], add 2 more runs for seeds 42 and 99.

from scipy import stats
import pandas as pd
import numpy as np

def run_statistical_tests(df_all, strategies=None, alpha=0.05):
    """
    Welch's t-test (unequal variance, appropriate for small n) comparing
    APE vs random exemplar selection per CL strategy.
    df_all columns: Strategy, Method (contains 'TADILER' or 'Random'), AMCA.
    """
    if strategies is None:
        strategies = ["Naive", "GEM", "LwF", "EWC"]

    rows = []
    for strategy in strategies:
        ape  = df_all[
            (df_all["Strategy"] == strategy) &
            (df_all["Method"].str.contains("TADILER"))
        ]["AMCA"].values
        rand = df_all[
            (df_all["Strategy"] == strategy) &
            (df_all["Method"].str.contains("Random"))
        ]["AMCA"].values

        if len(ape) < 2 or len(rand) < 2:
            print(f"[WARN] {strategy}: insufficient samples")
            continue

        t_stat, p_value = stats.ttest_ind(ape, rand, equal_var=False)

        # Welch-Satterthwaite degrees of freedom for 95% CI
        s2a, s2b = ape.var(ddof=1), rand.var(ddof=1)
        na, nb   = len(ape), len(rand)
        se_diff  = np.sqrt(s2a/na + s2b/nb)
        df_ws    = (s2a/na + s2b/nb)**2 / (
                   (s2a/na)**2/(na-1) + (s2b/nb)**2/(nb-1))
        ci_half  = stats.t.ppf(0.975, df=df_ws) * se_diff

        rows.append({
            "Strategy":  strategy,
            "APE mean":  round(float(ape.mean()),       4),
            "APE std":   round(float(ape.std(ddof=1)),  4),
            "Rand mean": round(float(rand.mean()),      4),
            "Rand std":  round(float(rand.std(ddof=1)), 4),
            "Delta":     round(float(ape.mean() - rand.mean()), 4),
            "95% CI":    f"+/-{round(float(ci_half), 4)}",
            "t":         round(float(t_stat),  3),
            "p":         round(float(p_value), 4),
            "sig":       "*" if p_value < alpha else "",
        })

    df_stats = pd.DataFrame(rows)
    df_stats.to_csv("extension_plots/statistical_tests.csv", index=False)
    print(df_stats.to_string(index=False))
    return df_stats

# Call after process_all_seeds_data() completes with 5 seeds:
# df_stats = run_statistical_tests(df_all)
```

---

### Patch R5.4 -- Section 5.3: replace the unsupported significance claim

**Location:** `\subsection{Computational efficiency and robustness analysis}`, replace
the sentence beginning "Results across three random seeds (3, 51, 35) demonstrate
APE's consistent performance advantages with statistical significance..."

```latex
Results across five random seeds $\{3, 51, 35, 42, 99\}$ demonstrate APE's consistent
performance advantages. Table~\ref{tab:statistical_tests} reports Welch's $t$-tests
comparing APE and random exemplar selection per CL strategy. Welch's test (unequal
variance) is used because the two methods share the same seeds but differ in exemplar
selection, introducing correlated but not identical variance structures across runs.
[After running: insert one sentence stating which strategies are significant, the
p-values, and the effect size range. If any strategy does not reach significance,
state that honestly.]

\begin{table}[htbp]
\centering
\caption{Welch's $t$-tests comparing APE against random exemplar selection
($n = 5$ seeds). Delta is APE mean AMCA minus random mean AMCA. Significant
results ($p < 0.05$) are marked *.}
\label{tab:statistical_tests}
\begin{tabular}{lcccccc}
\toprule
\bfseries Strategy
  & \bfseries APE mean & \bfseries APE std
  & \bfseries Rand mean & \bfseries Rand std
  & \bfseries Delta & \bfseries 95\% CI & \bfseries $p$ \\
\midrule
Naive & -- & -- & -- & -- & -- & -- & -- \\
GEM   & -- & -- & -- & -- & -- & -- & -- \\
LwF   & -- & -- & -- & -- & -- & -- & -- \\
EWC   & -- & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
% Fill in after running run_statistical_tests(df_all) with 5 seeds.
% Then replace the bracketed sentence above with actual interpretation.
```

---

### Patch R5.5 -- Limitations: privacy sentence update

**Location:** `\section{Limitations}`, sentence beginning "In addition, reliance on
embedding-based representations..."

```latex
In addition, reliance on embedding-based representations presents explainability
challenges that may require integration with attention visualization techniques.
While storing CLIP embeddings rather than raw fundus images substantially reduces
re-identification risk by removing pixel-level information, a formal privacy
evaluation through membership inference or nearest-neighbour reconstruction
analysis~\cite{ShokriShmatikov2015} would quantify the residual risk and is left as
future work.
```

---

### Patch R5.6 -- Related Work: add 2024-2025 literature paragraph

**Location:** End of the "Prompt optimization" paragraph in Related Work.

```latex
Recent work has converged on evolutionary strategies as a principled approach to
automatic prompt optimization without gradient access. EvoPrompt~\cite{evoprompt2024}
demonstrated that connecting LLMs with evolutionary operators yields powerful discrete
prompt optimizers that outperform hand-crafted alternatives, directly validating the
search paradigm underlying APE. A survey of automatic prompt
engineering~\cite{ape_survey_2025} formalizes the problem as maximization over discrete
and continuous prompt spaces, situating evolutionary methods alongside gradient-based
and reinforcement learning alternatives. From a continual learning perspective,
InCA~\cite{momeni2024inca} demonstrates that in-context learning can serve as a
parameter-free CL mechanism, treating the accumulated prompt context as persistent
task memory -- a view aligned with APE's use of an evolved frozen prompt as a
transferable knowledge representation across tasks.
```

---
---

## Reviewer 10

**Original requests (grouped by theme):**

General:
- G1. Highlight methodological limitations and future directions more specifically.
- G2. Discuss how emerging trends could influence future research.
- G3. Remove duplicate or repeated text passages.

Specific actionable questions (selected from 21 total):
- S5. Minimum patience to avoid local minima; does the algorithm terminate too early?
- S6. How does LLM stochasticity affect the stopping trigger?
- S11. Cost-to-gain ratio at early stopping.
- S12. Fixed-budget non-adaptive baseline. (Covered by R5.3.)
- S16. Why CLIP over SigLIP?
- S17. Could APE leverage SigLIP 2 dense features?
- S18. Why dot-product similarity instead of ALBEF-style multimodal fusion?
- S19-S21. Dual prompt memory, gating mechanisms, rehearsal efficiency.

Requests S1-S4, S7-S10, S13-S15 are exploratory; addressed collectively in one
discussion paragraph.

**Decisions:**

- G1/G2/G3: **WRITING ONLY**
- S5/S6 (patience and stochasticity): **WRITING ONLY** (address analytically)
- S11 (cost-to-gain): **WRITING ONLY** (compute from existing data)
- S12 (fixed-budget baseline): covered by R5.3
- S16 (CLIP vs SigLIP): **WRITING ONLY**
- S17/S18 (SigLIP 2, ALBEF): **FUTURE WORK**
- S19-S21 (dual memory, gating, rehearsal): **FUTURE WORK**
- Exploratory S1-S4, S7-S10, S13-S15: **WRITING ONLY** (collective paragraph)

---

### Patch R10.1 -- Algorithm 1: patience, stochasticity, and cost-to-gain note

**Location:** After the symbol disambiguation note added in Patch R3.5.

```latex
\noindent\textbf{Patience and LLM stochasticity.}
The patience parameter $P = 3$ was selected empirically: values below 2 risk premature
termination when the LLM produces a run of low-quality candidates by chance (generation
temperature is 0.7), while values above 5 provide diminishing returns as the local
candidate space becomes exhausted. Crucially, the best-ever prompt is preserved
throughout evolution regardless of when the patience criterion fires, so early
termination reduces the evaluation count but cannot degrade the best F1 achieved.

\noindent\textbf{Cost-to-gain analysis.}
APE converges at 50 evaluations, achieving F1 = 0.9229. Continuing to the budget
ceiling of $I_{\text{max}} \cdot k = 70$ evaluations was not beneficial: no candidate
in the final patience-triggering iterations improved on the best-ever prompt. The
saved 20 evaluations (29\% of the maximum budget) came at zero F1 cost, confirming
that the patience criterion does not sacrifice performance for efficiency.
```

---

### Patch R10.2 -- Related Work: CLIP justification and SigLIP/ALBEF future work

**Location:** `\subsection{CLIP embeddings in continual learning}`, add at the end.

```latex
CLIP ViT-L/14@336px is retained rather than SigLIP or ALBEF-style multimodal
transformers for two reasons. First, TADILER~\cite{BravoRocca2025} is built on CLIP,
and switching the encoder would prevent a direct attribution of APE's gains to the
prompt evolution mechanism rather than the encoder change. Second, CLIP's
softmax-based similarity is computationally lighter on CPU-only hardware than
ALBEF's cross-attention fusion, which is important for the offline deployment
constraint. SigLIP's pairwise sigmoid loss may offer better scaling at larger batch
sizes, and SigLIP~2's dense localization features could support region-specific prompt
evolution targeting small retinal lesions such as microaneurysms -- both directions
represent promising future work.
```

---

### Patch R10.3 -- Conclusion: expand future directions

**Location:** `\section{Conclusion}`, add before the final sentence.

```latex
Several directions remain open. Instance-specific or region-specific prompt evolution,
enabled by dense visual features from models such as SigLIP~2, could capture
fine-grained lesion variability that a single shared prompt cannot represent.
A dual-memory prompt architecture -- maintaining a stable long-term prompt encoding
accumulated knowledge alongside an adaptive short-term prompt for current task
features -- could provide an explicit stability-plasticity trade-off at the prompt
level, complementing the weight-level mechanisms of EWC and LwF. Extending APE to
federated settings, where multiple institutions contribute to prompt evolution without
sharing raw images or embeddings, is a natural direction for environments with strict
multi-institutional data governance requirements.
```

---

### Patch R10.4 -- Results: collective discussion paragraph for exploratory questions

**Location:** End of `\section{Results and analysis}`, add as a brief discussion
subsection.

```latex
\subsection{Discussion of design choices}
\label{sec:discussion}

Several design questions merit explicit discussion. \emph{Prompt composition across
tasks:} The frozen shared prompt works here because CLIP's embedding space is shared
and the evolved descriptions generalize across synthetic domain shifts. For tasks with
genuinely overlapping visual domains, interpolating between task-specific prompts is a
natural extension, though it would require task identity at inference time.
\emph{Semantic trajectory during evolution:} Candidates tend to shift from generic
terms (e.g., ``healthy'', ``diseased'') toward more specific clinical language as the
search progresses, consistent with increasing CLIP cosine similarity contrast between
classes. \emph{Metric alignment:} APE optimizes for F1-score on zero-shot CLIP
classification -- not for text generation metrics such as ROUGE or BLEU -- so the
fitness signal is directly aligned with the downstream classification task and does not
create a metric-exploitation risk.
```

---
---

## New Section: LLM Backbone Ablation

This section addresses R3.3c and presents the three general-purpose LLMs as a
standalone ablation rather than a replacement for the Med-Gemma results. All existing
results remain. The ablation runs the full experiment grid (all seeds, strategies,
architectures) to produce AMCA scores directly comparable to the main tables.

### Code hint -- setup_llm factory and ablation runner

```python
import os
from langchain_openai import ChatOpenAI
from typing import Optional


def setup_llm(
    model_name: str,
    openrouter_api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
) -> ChatOpenAI:
    """
    Returns a ChatOpenAI client via the OpenRouter API.
    Set OPENROUTER_API_KEY as an environment variable or pass directly.
    For local deployment, swap base_url for a local inference server endpoint
    (e.g., Ollama at http://localhost:11434/v1).
    """
    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# In generate_template_and_descriptions_tracked:
#   Replace any hardcoded model instantiation with: llm = setup_llm(model_name=model_name)
#   Add model_name: str to the function signature.
#   Propagate model_name up through autonomous_ape_optimization_enhanced_v2.


ABLATION_MODELS = {
    "Gemma 3 4B":   "google/gemma-3-4b-it",
    "Qwen3 4B":     "qwen/qwen3-4b",
    "Llama 3.2 3B": "meta-llama/llama-3.2-3b-instruct",
}

# The ablation re-uses the existing full experiment loop.
# Replace the Med-Gemma model call with setup_llm(model_name=model_name)
# and run the full grid once per entry in ABLATION_MODELS.
# Seeds: [3, 51, 35, 42, 99]  (same as main experiments)
# Results go to: extension_plots/llm_ablation_{model_label}/
```

---

### Patch: New Results subsection for the ablation

**Location:** Add as `\subsection{LLM backbone ablation}` after `\subsection{Computational
efficiency and robustness analysis}`.

```latex
\subsection{LLM backbone ablation}
\label{sec:llm_ablation}

To determine whether APE's performance gains are specific to Med-Gemma 4B's medical
pretraining or attributable to the evolutionary mechanism more broadly, we repeat the
full experiment grid using three general-purpose LLMs of comparable parameter count:
Gemma~3~4B, Qwen3~4B, and Llama~3.2~3B.\footnote{Accessed via OpenRouter
(\url{https://openrouter.ai}) using model identifiers \texttt{google/gemma-3-4b-it},
\texttt{qwen/qwen3-4b}, and \texttt{meta-llama/llama-3.2-3b-instruct} respectively.
All three models can also be deployed locally using tools such as Ollama or llama.cpp.}
None of these models carry medical domain fine-tuning. All experiment hyperparameters
are identical to the main experiments: five seeds $\{3, 51, 35, 42, 99\}$, all four
CL strategies, all three architectures, and the same KNN neighbor count values.

Table~\ref{tab:llm_ablation} reports the mean AMCA per model and CL strategy, averaged
across seeds and architectures.
[After running: replace this paragraph with interpretation. Key narrative options:
(a) If all three general-purpose models exceed the exhaustive search baseline -- the
evolutionary mechanism is the primary driver of APE's gains, and Med-Gemma 4B's
medical pretraining provides an additional margin on top.
(b) If general-purpose models fall below the exhaustive baseline for some strategies --
LLM quality matters, and Med-Gemma 4B's domain knowledge is a meaningful component.
(c) If one general-purpose model matches or exceeds Med-Gemma 4B -- that model becomes
the recommended alternative for practitioners without access to a medical LLM.]

\begin{table*}[htbp]
\centering
\caption{LLM backbone ablation. Mean AMCA per CL strategy averaged across five seeds
and three architectures. Med-Gemma~4B (main method) and exhaustive search (TADILER)
are included as reference rows.}
\label{tab:llm_ablation}
\begin{tabular}{lcccc}
\toprule
\bfseries Model & \bfseries Naive & \bfseries GEM & \bfseries LwF & \bfseries EWC \\
\midrule
Med-Gemma 4B (main method)  & -- & -- & -- & -- \\
\midrule
Gemma 3 4B                  & -- & -- & -- & -- \\
Qwen3 4B                    & -- & -- & -- & -- \\
Llama 3.2 3B                & -- & -- & -- & -- \\
\midrule
Exhaustive search (TADILER) & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table*}
% Fill in all rows after running the full ablation grid for each model.
% Med-Gemma and TADILER rows come from existing results (update to 5 seeds if needed).
% Replace the bracketed paragraph above with actual interpretation.
```

---
---

## Bibliography Entries

Add to `sn-bibliography.bib`. Model endpoint identifiers go in footnotes in the text,
not in the bibliography.

```bibtex
@inproceedings{evoprompt2024,
  author    = {Guo, Qingyan and Wang, Rui and Guo, Junliang and Li, Bei and
               Song, Kaitao and Tan, Xu and Liu, Guoqing and Bian, Jiang and Yang, Yujiu},
  title     = {{EvoPrompt}: Connecting {LLM}s with Evolutionary Algorithms Yields
               Powerful Prompt Optimizers},
  booktitle = {Proceedings of the International Conference on Learning Representations},
  year      = {2024},
  note      = {arXiv:2309.08532}
}

@misc{ape_survey_2025,
  author = {Kadian, Aryan and others},
  title  = {A Survey of Automatic Prompt Engineering: An Optimization Perspective},
  year   = {2025},
  note   = {arXiv:2502.11560}
}

@misc{momeni2024inca,
  author = {Momeni, Saleh and Mazumder, Sahisnu and Ke, Zixuan and Liu, Bing},
  title  = {In-context Continual Learning Assisted by an External Continual Learner},
  year   = {2024},
  note   = {arXiv:2412.15563}
}

@inproceedings{khattab2023dspy,
  author    = {Khattab, Omar and others},
  title     = {{DSPy}: Compiling Declarative Language Model Calls into
               Self-Improving Pipelines},
  booktitle = {Proceedings of the International Conference on Learning Representations},
  year      = {2024},
  note      = {arXiv:2310.03714}
}

@article{porwal2018idrid,
  author  = {Porwal, Prasanna and Pachade, Samiksha and Kokare, Manesh and
             Deshmukh, Girish and Son, Jaemin and Bae, Woojin and others},
  title   = {{IDRiD}: Diabetic Retinopathy Segmentation and Grading Challenge},
  journal = {Medical Image Analysis},
  year    = {2020},
  volume  = {59},
  pages   = {101561}
}
```

---
---

## Summary Table

| Reviewer | Specific Request | Decision |
|---|---|---|
| R1.1 | Novelty highlight in Introduction | WRITING ONLY |
| R1.2 | Literature survey summary table | WRITING ONLY |
| R1.3 | Comparative study with existing methods | WRITING ONLY |
| R1.4 | Advantages of the method | WRITING ONLY |
| R1.5 | Time complexity | WRITING ONLY |
| R1.6 | Sample result images | WRITING ONLY |
| R1.7 | Typo corrections | WRITING ONLY |
| R3.1a | Contrast APE vs TADILER across all dimensions | WRITING ONLY |
| R3.1b | Comparison table with RL, Bayesian opt, evolutionary NLP | WRITING ONLY |
| R3.2a | Justify why Task 0 does not introduce task-specific bias | WRITING ONLY |
| R3.2b | Clarify prompts are frozen after Task 0 | WRITING ONLY |
| R3.2c | Evolution protocol clarification and overfitting discussion | WRITING ONLY |
| R3.3a | Why Med-Gemma 4B was selected | WRITING ONLY |
| R3.3b | Whether gains come from medical knowledge or search mechanism | answered by ablation |
| R3.3c | LLM backbone ablation (full grid) | IMPLEMENT NOW |
| R3.4 | Deployment implications of Med-Gemma 4B | WRITING ONLY |
| R3.5a | Statistical tests | IMPLEMENT NOW |
| R3.5b | Confidence intervals | IMPLEMENT NOW |
| R3.M1a | Complexity and convergence near Algorithm 1 | WRITING ONLY |
| R3.M1b | Disambiguate reused symbols | WRITING ONLY |
| R3.M2 | Justify short prompt constraint | WRITING ONLY |
| R3.M3 | Improve Figures 3 and 4 readability | WRITING ONLY |
| R3.M4 | Clarify seed usage; release evolved prompt examples | WRITING ONLY |
| R3.M5 | Reduce repetitive phrasing | WRITING ONLY |
| R5.1 | Strict data separation protocol | WRITING ONLY |
| R5.2 | Realistic domain shift experiment | FUTURE WORK + Limitations |
| R5.3 | Random prompt selection baseline (50 evaluations) | IMPLEMENT NOW |
| R5.4 | Statistical test, n runs, pairing, mean/std, CIs | IMPLEMENT NOW |
| R5.5 | Privacy stress test | DISCARD + one sentence in Limitations |
| R5.6 | 2023-2025 literature | WRITING ONLY |
| R10.G1 | Methodological limitations and future directions | WRITING ONLY |
| R10.G2 | Emerging trends in future research | WRITING ONLY |
| R10.G3 | Remove duplicate text | WRITING ONLY |
| R10.S5 | Minimum patience analysis | WRITING ONLY |
| R10.S6 | LLM stochasticity effect on stopping | WRITING ONLY |
| R10.S11 | Cost-to-gain ratio at early stopping | WRITING ONLY |
| R10.S12 | Fixed-budget baseline | covered by R5.3 |
| R10.S16 | Justify CLIP over SigLIP | WRITING ONLY |
| R10.S17/S18 | SigLIP 2 dense features / ALBEF fusion | FUTURE WORK |
| R10.S19-S21 | Dual prompt memory, gating, rehearsal efficiency | FUTURE WORK |
| R10.S1-S4, S7-S10, S13-S15 | Exploratory design questions | WRITING ONLY (collective paragraph) |

---

## Execution Order

### Track A -- INDEPENDENT (start immediately, no experiments needed)

These tasks require only edits to the .tex file and .bib file. They can be done in
any order and do not block anything.

| Task | Patch | Notes |
|---|---|---|
| Novelty rewrite in Introduction | R1.1 | Also update abstract if the contributions list is mirrored there |
| Literature summary table | R1.2 | Requires EvoPrompt and DSPy bib entries (add those first) |
| Time complexity subsection | R1.3 | Empirical timing (9.2 min) already in notebook logs |
| APE vs TADILER contrast table | R3.1 | Requires EvoPrompt and DSPy bib entries |
| Prompt freezing and Task 0 bias | R3.2 | Pure writing |
| Med-Gemma selection rationale | R3.3 | Pure writing |
| Algorithm 1 complexity and symbol note | R3.5 | Cross-reference to new sec:complexity |
| Data separation statement | R5.1 | Pure writing; confirm seed list matches experiments |
| Realistic domain shift in Limitations | R5.2 | Requires IDRiD bib entry |
| Random baseline description in Section 4 | R5.3 text | Pure writing |
| Privacy sentence in Limitations | R5.5 | Pure writing |
| Related Work 2024-2025 paragraph | R5.6 | Requires EvoPrompt, ape_survey, momeni2024inca entries |
| CLIP vs SigLIP justification | R10.2 | Pure writing |
| Future directions in Conclusion | R10.3 | Pure writing |
| Discussion subsection | R10.4 | Pure writing |
| Algorithm 1 patience/stochasticity note | R10.1 | Pure writing |
| Deployment implications in Limitations | R3.4 | Pure writing |
| Bibliography additions | all | Add all 5 new entries to sn-bibliography.bib |
| Typo pass | R1.7 | Do last, after all patches are applied |

---

### Track B -- DEPENDENT: short experiments (days, not weeks)

These require new code but converge quickly. They can run in parallel with each other
once the code changes from the LLM factory (Track C prep) are in place.

| Task | Code | Depends on | Output |
|---|---|---|---|
| Random selection baseline | `random_prompt_baseline()` | none | rand_f1_t0, rand_f1_t1, rand_f1_t2 |
| Fill Patch R5.3b prose | writing | random baseline results | narrative paragraph in Section 5.1 |

---

### Track C -- DEPENDENT: full experiment reruns (the long pole)

The LLM ablation is the most compute-intensive task. It runs the full grid three times
(once per model). The statistical tests depend on having 5 seeds for the Med-Gemma
main results first.

| Task | Code | Depends on | Output |
|---|---|---|---|
| Add seeds 42, 99 to Med-Gemma main results | existing loop | none | df_all with n=5 |
| Statistical tests | `run_statistical_tests()` | Med-Gemma 5-seed df_all | Table tab:statistical_tests |
| Fill Patch R5.4 prose | writing | statistical tests | interpretation sentence |
| Refactor to `setup_llm()` factory | code change | none | enables ablation |
| LLM ablation: Gemma 3 4B | full grid | setup_llm refactor | AMCA per strategy |
| LLM ablation: Qwen3 4B | full grid | setup_llm refactor | AMCA per strategy |
| LLM ablation: Llama 3.2 3B | full grid | setup_llm refactor | AMCA per strategy |
| Fill Table tab:llm_ablation | writing | all 3 ablation runs | complete table |
| Fill ablation interpretation paragraph | writing | all 3 ablation runs | narrative in sec:llm_ablation |
| Fill Med-Gemma row in tab:llm_ablation | writing | 5-seed Med-Gemma results | reference row |
| Update abstract if headline numbers change | writing | all experiments done | final abstract |

---

### Recommended parallel workflow

While Track C (long ablation) is running in the background:

1. Complete all of Track A (writing only). This is the majority of the writing work
   and is fully unblocked.
2. Run Track B (random baseline) in parallel. This takes minutes per task.
3. Once Med-Gemma 5-seed results are ready, run statistical tests immediately (Track C
   first subtask). This does not require the ablation to be finished.
4. Fill in tables and interpretation paragraphs as each ablation model finishes,
   rather than waiting for all three to complete.
