import json
import os
import re
import time
from typing import Optional

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sklearn import metrics

from ape.clip_utils import get_label_tokens, get_zeroshot_sampling
from ape.tracker import APETracker

load_dotenv()

# ---------------------------------------------------------------------------
# Model registry for the LLM backbone ablation (R3.3c)
# ---------------------------------------------------------------------------
ABLATION_MODELS = {
    "Gemma3 4B":   "google/gemma-3-4b-it",
    "Granite4 3B": "ibm-granite/granite-4.0-h-micro",
    "Qwen3 30B":   "qwen/qwen3-30b-a3b-instruct-2507",
}


# ---------------------------------------------------------------------------
# Vocabulary for the budget-matched random baseline (R5.3)
# ---------------------------------------------------------------------------
# Generic templates — no anatomical or domain terminology.
# Goal: a truly uninformed random search whose only signal is
# the (template, healthy-word, diseased-word) composition itself.
RANDOM_TEMPLATES = [
    "", "Image shows", "Photo depicts",
    "Scan reveals", "Image depicts", "Picture shows",
    "Photo shows", "This image shows", "Visual inspection shows",
]
RANDOM_HEALTHY = [
    "healthy", "normal", "clear",
    "typical", "standard", "regular",
    "fine", "intact", "unremarkable", "unaffected",
]
RANDOM_DISEASED = [
    "diseased", "abnormal", "damaged",
    "pathological", "lesion", "affected",
    "impaired", "disordered", "altered", "compromised",
]


def setup_llm(
    model_name: str,
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1500,
) -> ChatOpenAI:
    """Generic LLM factory. Defaults to OpenRouter; override base_url for local endpoints.

    max_tokens defaults to 1500 (not 300) because thinking-mode models such as Qwen3
    can easily spend 300+ tokens in the <think> block before emitting the JSON response.
    """
    resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file or pass api_key explicitly."
        )
    return ChatOpenAI(
        base_url=base_url,
        api_key=resolved_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def setup_medgemma() -> ChatOpenAI:
    """Initialize Med-Gemma through LM Studio (local endpoint)."""
    return ChatOpenAI(
        base_url="http://172.27.144.1:1234/v1",
        api_key="lm-studio",
        model="med-gemma",
        temperature=0.7,
        max_tokens=300,
    )


def evaluate_single_prompt(template, descriptions, task_data, class_labels, embeder):
    """Evaluate a single template + description combination."""
    labels_prompts = [f"{template} {descriptions[c]}" for c in class_labels.values()]
    text_embs = get_label_tokens(embeder, labels_prompts)
    clustered_docs, sampled_docs = get_zeroshot_sampling(task_data, text_embs, labels_prompts, n_neighbors=10)

    ground_truth_labels = clustered_docs.label
    zeroshot_labels = clustered_docs.zeroshot_label
    f1 = metrics.f1_score(ground_truth_labels, zeroshot_labels, average='weighted')

    return f1


def generate_template_and_descriptions_tracked(
    current_template,
    current_descriptions,
    current_f1,
    num_candidates=5,
    llm: Optional[ChatOpenAI] = None,
    already_tried: Optional[set] = None,
):
    """Generate new template + description combinations using an LLM.

    Args:
        llm: Pre-built ChatOpenAI instance. If None, falls back to setup_medgemma().
        already_tried: Set of template strings already evaluated in this run.
            Passed to the LLM to prevent repeating the same candidates.
    """
    print(f"\n🤖 Generating {num_candidates} new template+description combinations...")
    print(f"Current best F1: {current_f1:.4f}")

    if llm is None:
        llm = setup_medgemma()

    avoid_block = ""
    if already_tried:
        tried_list = "\n".join(f'  - "{t}"' for t in sorted(already_tried))
        avoid_block = f"\nALREADY TRIED (do NOT repeat these templates):\n{tried_list}\n"

    prompt = f"""
You are a medical expert creating optimal text prompts for diabetic retinopathy classification using AI vision models.

CURRENT BEST (F1={current_f1:.3f}):
Template: "{current_template}"
Healthy: "{current_descriptions['No retinopathy']}"  
Diseased: "{current_descriptions['Retinopathy']}"
{avoid_block}
TASK: Generate {num_candidates} NEW and DIFFERENT template+description combinations that could beat F1={current_f1:.3f}

REQUIREMENTS:
- Include BOTH template and descriptions
- Keep ALL text SHORT (2-4 words max per description)  
- Use medical terminology that AI vision models understand
- Focus on visual features visible in fundus images
- Templates should help contextualize the descriptions
- Each combination must use a template NOT in the "already tried" list above

GOOD EXAMPLES:
- Template: "Fundus image shows" + Healthy: "Normal retina" + Diseased: "DR present"
- Template: "Retinal scan reveals" + Healthy: "Healthy fundus" + Diseased: "Diabetic damage"
- Template: "" + Healthy: "Clear vessels" + Diseased: "Retinal lesions"

Return EXACTLY this JSON format:
[
  {{"template": "template1", "healthy": "healthy1", "diseased": "diseased1"}},
  {{"template": "template2", "healthy": "healthy2", "diseased": "diseased2"}},
  {{"template": "template3", "healthy": "healthy3", "diseased": "diseased3"}},
  {{"template": "template4", "healthy": "healthy4", "diseased": "diseased4"}},
  {{"template": "template5", "healthy": "healthy5", "diseased": "diseased5"}}
]
"""

    try:
        response = llm.invoke(prompt)
        response_text = response.content

        # Strip thinking-mode wrappers emitted by models such as Qwen3 and DeepSeek-R1.
        # These blocks frequently contain '[' and ']' characters that break naive extraction.
        cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        cleaned = re.sub(r'<\|im_thinking\|>.*?<\|/im_thinking\|>', '', cleaned, flags=re.DOTALL)

        # Find the outer JSON array: locate the last '[' that opens a list of objects.
        # Using rfind on '[' is more robust when models emit prose before the JSON.
        start_idx = cleaned.rfind('[{')
        if start_idx == -1:
            start_idx = cleaned.find('[')
        end_idx = cleaned.rfind(']') + 1
        json_text = cleaned[start_idx:end_idx]

        combinations_list = json.loads(json_text)

        candidates = []
        for combo in combinations_list[:num_candidates]:
            template = combo.get('template', '').strip()
            healthy = combo.get('healthy', '').strip()
            diseased = combo.get('diseased', '').strip()

            if len(healthy.split()) <= 4 and len(diseased.split()) <= 4:
                candidates.append({
                    'template': template,
                    'descriptions': {
                        'No retinopathy': healthy,
                        'Retinopathy': diseased
                    }
                })

        print(f"Generated {len(candidates)} valid combinations")
        return candidates

    except Exception as e:
        print(f"Med-Gemma generation failed: {e}")
        return []


def evaluate_template_combinations_tracked(candidates, task_data, current_f1, tracker, iteration, class_labels, embeder, llm=None):
    """Evaluate all template+description candidates with tracking."""
    print(f"\n📊 Evaluating {len(candidates)} template+description combinations...")

    evaluations = []
    for candidate in candidates:
        f1_score = evaluate_single_prompt(candidate['template'], candidate['descriptions'], task_data, class_labels, embeder)
        evaluations.append(f1_score)

    tracker.add_candidates(iteration, candidates, evaluations)

    best_f1 = current_f1
    best_template = None
    best_descriptions = None
    improved = False
    best_candidate_idx = -1

    for i, (candidate, f1_score) in enumerate(zip(candidates, evaluations)):
        template = candidate['template']
        descriptions = candidate['descriptions']

        print(f"  Candidate {i+1}: F1={f1_score:.4f} | Template: '{template}' | {descriptions}")

        if f1_score > best_f1:
            print(f"  🎉 NEW BEST! Improvement: +{f1_score - best_f1:.4f}")
            best_f1 = f1_score
            best_template = template
            best_descriptions = descriptions
            improved = True
            best_candidate_idx = i

    iteration_stats = {
        'num_candidates': len(candidates),
        'best_f1': max(evaluations),
        'worst_f1': min(evaluations),
        'avg_f1': sum(evaluations) / len(evaluations),
        'std_f1': (sum([(f1 - sum(evaluations)/len(evaluations))**2 for f1 in evaluations]) / len(evaluations))**0.5,
        'improvements': sum(1 for f1 in evaluations if f1 > current_f1),
        'best_candidate_idx': best_candidate_idx
    }
    tracker.add_iteration_stats(iteration, iteration_stats)

    return best_f1, best_template, best_descriptions, improved


def autonomous_ape_optimization_enhanced_v2(
    task_0_data,
    class_labels,
    embeder,
    max_iterations=7,
    candidates_per_iteration=5,
    target_score=None,
    patience=3,
    starting_template="",
    starting_descriptions=None,
    baseline_scores=None,
    verbose=True,
    stop_on_target=False,
    # LLM backbone selection (leave None to use Med-Gemma via LM Studio)
    model_name: Optional[str] = None,
    llm_base_url: str = "https://openrouter.ai/api/v1",
    llm_api_key: Optional[str] = None,
):
    """
    Enhanced autonomous APE optimization with option to continue past target for better generalization.

    Args:
        task_0_data: Task 0 dataset with ground truth
        class_labels: Dict mapping class indices to label strings
        embeder: CLIP model for encoding text
        max_iterations: Maximum evolution cycles
        candidates_per_iteration: Number of candidates per iteration
        target_score: Target F1 score to track
        patience: Iterations without improvement before early stopping
        starting_template: Initial template (default: "")
        starting_descriptions: Initial descriptions
        baseline_scores: Dictionary of baseline scores for comparison
        verbose: Whether to print detailed progress
        stop_on_target: If True, stops when target is achieved
        model_name: OpenRouter model ID for ablation; None uses Med-Gemma via LM Studio
        llm_base_url: LLM API base URL (default: OpenRouter)
        llm_api_key: API key override; reads OPENROUTER_API_KEY from env if None
    """
    # Build LLM once so it is reused across all iterations
    if model_name is not None:
        _llm = setup_llm(
            model_name=model_name,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
    else:
        _llm = setup_medgemma()

    tracker = APETracker()

    if verbose:
        print("🚀 Starting Enhanced Autonomous APE Optimization v2")
        print("=" * 60)

    if starting_descriptions is None:
        starting_descriptions = {'No retinopathy': 'healthy', 'Retinopathy': 'diseased'}

    if baseline_scores is None:
        baseline_scores = {'exhaustive_search': 0.8921}

    current_template = starting_template
    current_descriptions = starting_descriptions.copy()
    tried_templates: set = {starting_template}  # track to avoid re-generating the same templates

    if verbose:
        print("🌱 Evaluating starting point...")
    current_f1 = evaluate_single_prompt(current_template, current_descriptions, task_0_data, class_labels, embeder)

    initial_state = {
        'template': current_template,
        'descriptions': current_descriptions,
        'f1': current_f1
    }
    tracker.start_tracking(initial_state, target_score, baseline_scores)

    if verbose:
        print(f"\n📍 STARTING CONFIGURATION:")
        print(f"  Template: '{current_template}' {'(empty)' if not current_template else ''}")
        print(f"  Descriptions: {current_descriptions}")
        print(f"  Starting F1: {current_f1:.4f}")
        if target_score:
            print(f"  🎯 Target: Beat F1 > {target_score:.4f}")
            print(f"  🔄 Stop on target: {'Yes' if stop_on_target else 'No (continue for generalization)'}")
        print(f"  🔄 Max iterations: {max_iterations}")
        print(f"  ⏱️ Patience: {patience} iterations")

    consecutive_no_improvement = 0
    target_achieved_iteration = None

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n" + "="*50)
            print(f"🧬 EVOLUTION ITERATION {iteration + 1}/{max_iterations}")
            print(f"="*50)
            print(f"Current best F1: {current_f1:.4f}")

        if target_score and current_f1 >= target_score:
            if target_achieved_iteration is None:
                target_achieved_iteration = iteration + 1
                if verbose:
                    print(f"🎯 TARGET ACHIEVED! F1={current_f1:.4f} >= {target_score:.4f}")
                    if stop_on_target:
                        print("🛑 Stopping early due to target achievement")
                        break
                    else:
                        print("🔄 Continuing evolution for better generalization...")

        candidates = generate_template_and_descriptions_tracked(
            current_template, current_descriptions, current_f1,
            num_candidates=candidates_per_iteration,
            llm=_llm,
            already_tried=tried_templates,
        )

        if not candidates:
            if verbose:
                print("❌ No valid candidates generated, stopping early")
            break

        # Register all generated templates so the next iteration avoids them
        for c in candidates:
            tried_templates.add(c["template"])

        new_f1, new_template, new_descriptions, improved = evaluate_template_combinations_tracked(
            candidates, task_0_data, current_f1, tracker, iteration + 1, class_labels, embeder, llm=_llm
        )

        if improved:
            previous_f1 = current_f1
            current_f1 = new_f1
            current_template = new_template
            current_descriptions = new_descriptions
            consecutive_no_improvement = 0

            improvement = current_f1 - previous_f1

            if verbose:
                print(f"\n🎉 EVOLUTION SUCCESS!")
                print(f"   Previous F1: {previous_f1:.4f}")
                print(f"   New F1: {current_f1:.4f}")
                print(f"   Improvement: +{improvement:.4f}")
                print(f"   New template: '{current_template}'")
                print(f"   New descriptions: {current_descriptions}")

            tracker.add_evolution_step(iteration + 1, {
                'template': current_template,
                'descriptions': current_descriptions,
                'f1': current_f1
            }, improvement)

        else:
            consecutive_no_improvement += 1
            if verbose:
                print(f"\n📊 No improvement in iteration {iteration + 1}")
                print(f"   Consecutive no improvement: {consecutive_no_improvement}")
                print(f"   Current best remains: {current_f1:.4f}")

            if consecutive_no_improvement >= patience:
                if verbose:
                    print(f"\n🛑 Early stopping: No improvement for {patience} iterations")
                break

    tracker.finish_tracking()
    tracking_data = tracker.get_summary()
    tracking_data['summary']['target_achieved_iteration'] = target_achieved_iteration

    if verbose:
        print(f"\n" + "="*60)
        print("🧬 ENHANCED APE EVOLUTION COMPLETE")
        print("="*60)

        print("📈 Evolution Trajectory:")
        for step in tracking_data['evolution_history']:
            if step['iteration'] == 0:
                print(f"  Start: F1={step['f1']:.4f} | '{step['template']}' + {step['descriptions']}")
            else:
                print(f"  Step {step['iteration']}: F1={step['f1']:.4f} (+{step['improvement']:.4f}) | '{step['template']}' + {step['descriptions']}")

        summary = tracking_data['summary']
        print(f"\n📊 Final Statistics:")
        print(f"  Total candidates tested: {summary['total_candidates']}")
        print(f"  Evolution steps: {summary['evolution_steps']}")
        print(f"  Starting F1: {summary['starting_f1']:.4f}")
        print(f"  Final F1: {summary['final_f1']:.4f}")
        print(f"  Total improvement: +{summary['total_improvement']:.4f}")
        print(f"  Total time: {summary['total_time']:.1f} seconds")

        if target_achieved_iteration:
            print(f"  🎯 Target achieved at iteration: {target_achieved_iteration}")
            additional_iterations = len(tracking_data['evolution_history']) - 1 - target_achieved_iteration
            if additional_iterations > 0:
                print(f"  🔄 Additional iterations for generalization: {additional_iterations}")

        if baseline_scores:
            print(f"\n🏆 Baseline Comparisons:")
            for name, score in baseline_scores.items():
                if summary['final_f1'] >= score:
                    print(f"  ✅ {name}: {score:.4f} → APE beats by +{summary['final_f1'] - score:.4f}")
                else:
                    print(f"  📈 {name}: {score:.4f} → APE gap: -{score - summary['final_f1']:.4f}")

        if target_score:
            if summary['target_achieved']:
                print(f"  🎯 Target {target_score:.4f}: ✅ ACHIEVED!")
            else:
                print(f"  🎯 Target {target_score:.4f}: ❌ Gap: -{target_score - summary['final_f1']:.4f}")

    return {
        'template': current_template,
        'descriptions': current_descriptions,
        'f1_score': current_f1,
        'tracking_data': tracking_data
    }


def run_enhanced_autonomous_ape_v2(
    list_doclists,
    class_labels,
    embeder,
    max_iterations=7,
    candidates_per_iteration=5,
    target_score=0.8921,
    patience=3,
    starting_template="",
    starting_descriptions=None,
    baseline_scores=None,
    stop_on_target=False
):
    """Run enhanced autonomous APE optimization for all tasks."""
    print("🚀 Starting Enhanced Autonomous APE v2 for All Tasks")
    print("🧬 Learning generalizable prompts (no early target stopping)")
    print("="*70)

    if baseline_scores is None:
        baseline_scores = {
            'exhaustive_search': 0.8921,
            'exhaustive_task1': 0.8899,
            'exhaustive_task2': 0.8909
        }

    ape_results = {}

    for task_index, task in enumerate(list_doclists):
        print(f"\n" + "="*60)
        print(f"PROCESSING TASK {task_index}")
        print(f"="*60)

        if task_index == 0:
            print("🧬 Running enhanced autonomous evolution on Task 0...")
            print("🔄 Will continue past target for better generalization...")

            ape_result = autonomous_ape_optimization_enhanced_v2(
                task,
                class_labels=class_labels,
                embeder=embeder,
                max_iterations=max_iterations,
                candidates_per_iteration=candidates_per_iteration,
                target_score=target_score,
                patience=patience,
                starting_template=starting_template,
                starting_descriptions=starting_descriptions,
                baseline_scores=baseline_scores,
                stop_on_target=stop_on_target
            )

            ape_results[task_index] = ape_result

        else:
            print(f"📋 Transferring learned prompt from Task 0 to Task {task_index}...")
            learned_prompt = ape_results[0]

            f1_score = evaluate_single_prompt(
                learned_prompt['template'],
                learned_prompt['descriptions'],
                task,
                class_labels,
                embeder
            )

            ape_results[task_index] = {
                'template': learned_prompt['template'],
                'descriptions': learned_prompt['descriptions'],
                'f1_score': f1_score,
                'tracking_data': None
            }

            print(f"✅ Task {task_index} Results:")
            print(f"   Transfer F1: {f1_score:.4f}")

            baseline_key = f'exhaustive_task{task_index}'
            if baseline_key in baseline_scores:
                baseline_f1 = baseline_scores[baseline_key]
                if f1_score >= baseline_f1:
                    print(f"   🎉 Beats baseline: +{f1_score - baseline_f1:.4f}")
                else:
                    print(f"   📈 Below baseline: -{baseline_f1 - f1_score:.4f}")

    best_values = {}
    for task_index, result in ape_results.items():
        best_values[task_index] = {
            "best_score": result['f1_score'],
            "best_template": result['template'],
            "best_description_set": result['descriptions']
        }

    best_values['ape_tracking'] = ape_results[0]['tracking_data']

    print(f"\n" + "="*70)
    print("🎯 GENERALIZATION ANALYSIS")
    print("="*70)

    task_0_f1 = ape_results[0]['f1_score']
    transfer_f1s = [ape_results[i]['f1_score'] for i in range(1, len(list_doclists))]
    avg_transfer_f1 = sum(transfer_f1s) / len(transfer_f1s) if transfer_f1s else 0

    print(f"📊 Performance Summary:")
    print(f"  Task 0 (learning): {task_0_f1:.4f}")
    print(f"  Average transfer: {avg_transfer_f1:.4f}")
    print(f"  Transfer gap: {task_0_f1 - avg_transfer_f1:.4f}")

    baseline_transfers = [baseline_scores.get(f'exhaustive_task{i}', 0.85) for i in range(1, len(list_doclists))]
    avg_baseline_transfer = sum(baseline_transfers) / len(baseline_transfers) if baseline_transfers else 0

    print(f"\n🏆 Transfer vs Baseline:")
    print(f"  APE average transfer: {avg_transfer_f1:.4f}")
    print(f"  Baseline average transfer: {avg_baseline_transfer:.4f}")
    if avg_transfer_f1 >= avg_baseline_transfer:
        print(f"  ✅ APE transfer beats baseline: +{avg_transfer_f1 - avg_baseline_transfer:.4f}")
    else:
        print(f"  📈 APE transfer below baseline: -{avg_baseline_transfer - avg_transfer_f1:.4f}")

    return best_values


# ---------------------------------------------------------------------------
# Budget-matched random prompt baseline (R5.3)
# ---------------------------------------------------------------------------

def random_prompt_baseline(
    task_data,
    class_labels: dict,
    embeder,
    n_evaluations: int = 50,
    seed: int = 42,
):
    """Budget-matched random search over fixed vocabularies.

    Samples template + healthy/diseased descriptions uniformly at random using the
    same 50-evaluation budget as APE.  Returns the best combination found, the best
    F1 score, and the full trajectory of F1 scores.

    Args:
        task_data: DocList for the target task (used by evaluate_single_prompt).
        class_labels: Dict mapping class indices to label strings, e.g.
            {0: 'No retinopathy', 1: 'Retinopathy'}.
        embeder: CLIP model instance.
        n_evaluations: Number of random combinations to evaluate (default 50).
        seed: RNG seed for reproducibility.

    Returns:
        (best_combo, best_f1, all_f1s)
        best_combo: dict with keys 'template' and 'descriptions'.
        best_f1: float.
        all_f1s: list of floats, one per evaluation.
    """
    rng = np.random.default_rng(seed)
    best_f1, best_combo, all_f1s = 0.0, None, []

    for i in range(n_evaluations):
        template    = str(rng.choice(RANDOM_TEMPLATES))
        healthy     = str(rng.choice(RANDOM_HEALTHY))
        diseased    = str(rng.choice(RANDOM_DISEASED))
        descriptions = {
            list(class_labels.values())[0]: healthy,
            list(class_labels.values())[1]: diseased,
        }
        f1 = evaluate_single_prompt(template, descriptions, task_data, class_labels, embeder)
        all_f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_combo = {"template": template, "descriptions": descriptions}

        if (i + 1) % 10 == 0:
            print(f"  [Random baseline] eval {i+1}/{n_evaluations}  "
                  f"best={best_f1:.4f}  avg={float(np.mean(all_f1s)):.4f}")

    print(f"\n[Random baseline] DONE  best={best_f1:.4f}  avg={float(np.mean(all_f1s)):.4f}")
    return best_combo, best_f1, all_f1s


# ---------------------------------------------------------------------------
# LLM backbone ablation (R3.3c)
# ---------------------------------------------------------------------------

def run_llm_ablation(
    list_doclists,
    class_labels: dict,
    embeder,
    openrouter_api_key: Optional[str] = None,
    max_iterations: int = 7,
    n_candidates: int = 5,
    patience: int = 3,
    output_csv: str = "extension_plots/llm_ablation.csv",
) -> dict:
    """Run APE evolution once per model on Task 0, then transfer frozen prompt.

    Each model evolves on Task 0 only.  The resulting prompt is applied without
    modification to Tasks 1 and 2.  No CL strategies, no architecture variants,
    no seed loops.

    Args:
        list_doclists: List of DocLists, one per task.
        class_labels: Dict mapping class indices to label strings.
        embeder: CLIP model instance.
        openrouter_api_key: Optional API key override; reads OPENROUTER_API_KEY
            from environment (via .env) if None.
        max_iterations: Evolution iterations per model (default 7, same as main run).
        n_candidates: Candidates per iteration (default 5).
        patience: Early-stopping patience (default 3).
        output_csv: Path to save results CSV.

    Returns:
        Dict keyed by model label, each value containing f1_task0..N,
        template, and descriptions.
    """
    results = {}

    for label, model_name in ABLATION_MODELS.items():
        print(f"\n{'='*60}")
        print(f"APE ablation — {label}  ({model_name})")
        print(f"{'='*60}")

        result = autonomous_ape_optimization_enhanced_v2(
            task_0_data=list_doclists[0],
            class_labels=class_labels,
            embeder=embeder,
            max_iterations=max_iterations,
            candidates_per_iteration=n_candidates,
            patience=patience,
            target_score=None,
            stop_on_target=False,
            model_name=model_name,
            llm_api_key=openrouter_api_key,
        )

        row = {
            "f1_task0":    result["f1_score"],
            "template":    result["template"],
            "descriptions": result["descriptions"],
        }
        for i in range(1, len(list_doclists)):
            row[f"f1_task{i}"] = evaluate_single_prompt(
                result["template"],
                result["descriptions"],
                list_doclists[i],
                class_labels,
                embeder,
            )

        results[label] = row
        n_tasks = len(list_doclists)
        task_scores = "  ".join(f"T{i}={row[f'f1_task{i}']:.4f}" for i in range(n_tasks))
        print(f"  → {task_scores}")
        print(f"  Template: '{result['template']}' | Descriptions: {result['descriptions']}")

    # Persist results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    score_cols = [f"f1_task{i}" for i in range(len(list_doclists))]
    df = pd.DataFrame(results).T[score_cols]
    df.index.name = "Model"
    df.to_csv(output_csv)
    print(f"\nAblation results saved to {output_csv}")
    return results
