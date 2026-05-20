import time


class APETracker:
    """Tracks all APE evolution data for analysis and plotting."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.evolution_history = []
        self.all_candidates = []
        self.iteration_stats = []
        self.start_time = None
        self.end_time = None
        self.target_score = None
        self.baseline_scores = {}

    def start_tracking(self, initial_state, target_score=None, baseline_scores=None):
        self.start_time = time.time()
        self.target_score = target_score
        self.baseline_scores = baseline_scores or {}

        self.evolution_history.append({
            'iteration': 0,
            'template': initial_state['template'],
            'descriptions': initial_state['descriptions'].copy(),
            'f1': initial_state['f1'],
            'improvement': 0.0,
            'is_best': True,
            'timestamp': time.time() - self.start_time
        })

    def add_candidates(self, iteration, candidates, evaluations):
        for i, (candidate, f1_score) in enumerate(zip(candidates, evaluations)):
            self.all_candidates.append({
                'iteration': iteration,
                'candidate_id': i + 1,
                'template': candidate['template'],
                'descriptions': candidate['descriptions'].copy(),
                'f1': f1_score,
                'improvement': f1_score - self.evolution_history[-1]['f1'],
                'timestamp': time.time() - self.start_time
            })

    def add_evolution_step(self, iteration, new_state, improvement):
        self.evolution_history.append({
            'iteration': iteration,
            'template': new_state['template'],
            'descriptions': new_state['descriptions'].copy(),
            'f1': new_state['f1'],
            'improvement': improvement,
            'is_best': True,
            'timestamp': time.time() - self.start_time
        })

    def add_iteration_stats(self, iteration, stats):
        stats['iteration'] = iteration
        stats['timestamp'] = time.time() - self.start_time
        self.iteration_stats.append(stats)

    def finish_tracking(self):
        self.end_time = time.time()

    def get_summary(self):
        total_time = (self.end_time - self.start_time) if self.end_time else 0

        return {
            'evolution_history': self.evolution_history,
            'all_candidates': self.all_candidates,
            'iteration_stats': self.iteration_stats,
            'summary': {
                'total_time': total_time,
                'total_candidates': len(self.all_candidates),
                'evolution_steps': len(self.evolution_history) - 1,
                'starting_f1': self.evolution_history[0]['f1'],
                'final_f1': self.evolution_history[-1]['f1'],
                'total_improvement': self.evolution_history[-1]['f1'] - self.evolution_history[0]['f1'],
                'target_score': self.target_score,
                'target_achieved': self.evolution_history[-1]['f1'] >= (self.target_score or 0),
                'baseline_scores': self.baseline_scores
            }
        }
