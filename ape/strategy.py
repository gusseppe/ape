from avalanche.training.supervised import Replay, GEM, LwF, EWC, Naive
from torch.nn import CrossEntropyLoss


def get_sequence(sequence, stream):
    """Return a reordered list of experiences from a stream."""
    return [stream[i] for i in sequence]


def get_strategy(strategy_name, model, optimizer, criterion, eval_plugin, n_epochs, custom_replay, device='cpu'):
    """Instantiate an Avalanche continual learning strategy by name.

    Args:
        strategy_name: One of 'Naive', 'EWC', 'Replay', 'LwF', 'GEM'
        model: Multi-task model
        optimizer: PyTorch optimizer
        criterion: Loss function
        eval_plugin: Avalanche EvaluationPlugin
        n_epochs: Training epochs per experience
        custom_replay: TADILER plugin instance
        device: 'cpu' or 'cuda'
    """
    strategies = {
        'Naive': Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=200,
            train_epochs=n_epochs,
            eval_mb_size=200,
            device=device,
            evaluator=eval_plugin,
            plugins=[custom_replay],
        ),
        'EWC': EWC(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=200,
            train_epochs=n_epochs,
            eval_mb_size=200,
            device=device,
            evaluator=eval_plugin,
            ewc_lambda=0.2,
            plugins=[custom_replay],
        ),
        'Replay': Replay(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=200,
            train_epochs=n_epochs,
            eval_mb_size=200,
            device=device,
            evaluator=eval_plugin,
            plugins=[custom_replay],
            mem_size=20,
        ),
        'LwF': LwF(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            alpha=0.5,
            temperature=0.2,
            train_epochs=n_epochs,
            device=device,
            train_mb_size=200,
            eval_mb_size=200,
            evaluator=eval_plugin,
            plugins=[custom_replay],
        ),
        'GEM': GEM(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            patterns_per_exp=10,
            train_epochs=n_epochs,
            device=device,
            train_mb_size=200,
            eval_mb_size=200,
            evaluator=eval_plugin,
            plugins=[custom_replay],
        ),
    }

    return strategies[strategy_name]
