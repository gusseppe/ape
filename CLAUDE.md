# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Adaptive Prompt Evolution (APE)** — a research framework combining evolutionary prompt optimization with continual learning for diabetic retinopathy detection. Uses CLIP (vision-language model) to generate dynamic text prompts that reduce catastrophic forgetting across sequential learning tasks.

The implementation is entirely **notebook-based** — there are no standalone `.py` scripts. All research code lives in:
- `tadiler_dev13_APE_dynamic_prompt.ipynb` — main implementation (APE evolution + TADILER continual learning experiments)
- `tadiler_extension.ipynb` — post-experiment analysis and LaTeX table generation

## Environment Setup

Uses **uv** as the package manager (not pip/conda). Python 3.10 is pinned via `.python-version`.

```bash
uv sync                        # create .venv/ and install all dependencies
.venv/bin/python -m ipykernel install --user --name ape --display-name "APE (uv)"
```

Adding/removing packages:
```bash
uv add <package>
uv remove <package>
```

An **OpenAI API key** is required for the LangChain/OpenAI prompt generation steps in the notebook. Set `OPENAI_API_KEY` in your environment before running.

## Architecture

### APE Framework (3 core components)
1. **Dynamic Template Generation** — generates/evaluates diverse prompt templates via LLM (OpenAI via LangChain)
2. **Semantic Description Evolution** — optimizes medical class descriptions using F1-score feedback
3. **Performance-Driven Selection** — selects best prompts to guide evolutionary improvements

### TADILER Framework
Combines APE with Avalanche-based continual learning strategies:
- **Naive**, **EWC**, **Experience Replay**, **LwF**, **GEM**
- 3 neural architectures: **Attention**, **Residual**, **SMLP**
- Dataset: APTOS 2019 (3,662 retinal images, 3 sequential tasks)

### Key Dependencies
| Package | Role |
|---|---|
| `clip` (from GitHub) | Vision-language embeddings for prompt evaluation |
| `avalanche-lib` | Continual learning strategies (EWC, GEM, Replay, LwF) |
| `langchain-openai` / `openai` | LLM-based prompt generation and evolution |
| `docarray[torch]` | `BaseDoc`/`DocList` data structures for embeddings |
| `proxsuite` | Required by Avalanche's GEM solver (pinned ≤0.7.2 for wheel availability) |
| `umap-learn` | Dimensionality reduction for cluster visualizations |

## Results and Outputs

Experimental results are stored in `extension_plots/`:
- `df_results_all.csv` / `df_results_{attention,residual,smlp}.csv` — aggregated accuracy metrics
- `results-attention/`, `results-residual/`, `results-smlp/` — per-run `.pickle` checkpoints
- `best_values_tracking_v*.pickle` — APE prompt evolution state snapshots
- PNG/PDF plots for all figures

## Notebook Workflow

Main notebook sections run sequentially:
1. Data loading (APTOS 2019 dataset — path must be configured)
2. CLIP embedding + zero-shot cluster visualization
3. APE prompt evolution loop (saves `best_values_tracking_v*.pickle`)
4. TADILER continual learning experiments (saves per-strategy results to `results-*/`)
5. Result aggregation → CSV and LaTeX tables

The extension notebook reads the saved CSVs/pickles and regenerates plots without re-running experiments.
