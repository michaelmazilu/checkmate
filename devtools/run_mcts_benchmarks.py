import importlib.util
import random
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcts import MCTS as NewMCTS


def load_previous_mcts():
    """Load the previous MCTS implementation from the saved snapshot."""
    prev_path = Path(__file__).with_name("mcts_prev.py")
    spec = importlib.util.spec_from_file_location("mcts_prev", prev_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load previous MCTS implementation.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MCTS


def make_uniform_nn(counter):
    """Create a uniform neural-net stub that tracks how many times it's called."""

    def _nn(fen: str):
        counter["calls"] += 1
        board = chess.Board(fen)
        moves = list(board.generate_legal_moves())
        if not moves:
            return {}, 0.0
        prob = 1.0 / len(moves)
        return {move: prob for move in moves}, 0.0

    return _nn


def run_mcts(mcts_cls, name, num_simulations=1000):
    board = chess.Board()
    counter = {"calls": 0}
    nn = make_uniform_nn(counter)
    random.seed(0)
    start = time.perf_counter()
    mcts = mcts_cls(board, nn, num_simulations=num_simulations, c_puct=1.5)
    mcts.search()
    elapsed = time.perf_counter() - start
    best_move = mcts.get_best_move()
    visits = sum(child.visit_count for child in mcts.root.children.values())
    return {
        "name": name,
        "time": elapsed,
        "nn_calls": counter["calls"],
        "visits": visits,
        "best_move": str(best_move) if best_move else None,
    }


def main():
    prev_cls = load_previous_mcts()
    baseline = run_mcts(prev_cls, "baseline")
    optimized = run_mcts(NewMCTS, "optimized")
    print("MCTS benchmark comparison (start position, 1000 simulations):")
    for result in (baseline, optimized):
        print(
            f"- {result['name']:10s}: {result['time']:.3f}s, "
            f"NN calls={result['nn_calls']}, visits={result['visits']}, "
            f"best move={result['best_move']}"
        )
    nn_delta = baseline["nn_calls"] - optimized["nn_calls"]
    time_delta = baseline["time"] - optimized["time"]
    print("\nDelta (baseline - optimized):")
    print(f"- NN calls reduced by {nn_delta}")
    print(f"- Time reduced by {time_delta:.3f}s")


if __name__ == "__main__":
    main()
