from .utils import chess_manager, GameContext
from .mcts import MCTS
from chess import Move
import time
import chess
import os
from pathlib import Path

# ============================================================================
# NEURAL NETWORK SETUP
# ============================================================================

# Configuration
HF_REPO_ID = "raf-fonseca/checkmate-chess"
HF_MODEL_FILENAME = "checkmate_model.pt"

# Try to load the trained model from Hugging Face
USE_TRAINED_MODEL = False
model = None

print(f"[STARTUP] 🤗 Loading model from Hugging Face: {HF_REPO_ID}")
print(f"[STARTUP] 📦 Model file: {HF_MODEL_FILENAME}")

try:
    import sys
    from huggingface_hub import hf_hub_download
    
    print(f"[STARTUP] 🔄 Downloading model from Hugging Face...")
    print(f"[STARTUP] Repository: https://huggingface.co/{HF_REPO_ID}")
    
    # Download model from Hugging Face (will use cached version if already downloaded)
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        cache_dir=Path(__file__).parent.parent / ".model_cache"
    )
    
    print(f"[STARTUP] ✓ Model downloaded/loaded from Hugging Face!")
    print(f"[STARTUP] 📂 Cached at: {model_path}")
    
    # Load inference module
    sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
    from inference import ChessModelInference
    
    # Initialize model with the HF downloaded path
    print(f"[STARTUP] 🧠 Initializing neural network...")
    model = ChessModelInference(model_path, device="cpu")
    USE_TRAINED_MODEL = True
    print(f"[STARTUP] ✓ Model loaded successfully from Hugging Face checkpoint!")
    print(f"[STARTUP] 🚀 Ready to play with trained model!")
    
except Exception as e:
    print(f"[STARTUP] ⚠ Failed to load model from Hugging Face: {e}")
    print(f"[STARTUP] Error type: {type(e).__name__}")
    print("[STARTUP] Falling back to hardcoded neural network")
    print("[STARTUP] Using uniform priors (P={}) and neutral value (V=0)")
    USE_TRAINED_MODEL = False


def neural_net(board_fen: str):
    """
    Returns policy priors (P) and position evaluation (V) for a given board state.
    
    Args:
        board_fen: Board state in FEN format
        
    Returns:
        P: Dict mapping moves (as Move objects) to probabilities (0-1)
           These represent the NN's prior belief about which moves are good
        V: Float in range [-1, 1] representing position evaluation
           Positive = current player winning, Negative = current player losing
    """
    if USE_TRAINED_MODEL:
        # Use trained neural network
        board = chess.Board(board_fen)
        P, V = model.predict(board_fen, board)
        print(f"[NN] Trained model -> P: {len(P)} moves, V: {V:.3f}")
        return P, V
    else:
        # Hardcoded fallback (uniform distribution)
        P = {}
        V = 0.0
        print(f"[NN] Hardcoded -> P: {len(P)} moves, V: {V:.3f}")
        return P, V


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """Main inference function. Uses MCTS to select the best move."""
    print("Running MCTS search...")
    start_time = time.time()
    
    # Initialize MCTS and run search
    mcts = MCTS(ctx.board, neural_net, num_simulations=100, c_puct=1.0)
    mcts.search()
    
    elapsed = time.time() - start_time
    print(f"Search completed in {elapsed:.2f}s")
    
    # Get best move and move probabilities
    best_move = mcts.get_best_move()
    move_probs = mcts.get_move_probabilities()
    
    if best_move is None:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Log the move probabilities for analysis (using Move objects as keys)
    ctx.logProbabilities(move_probs)
    
    # Print top 5 moves by visit count
    print("\n[MCTS Results] Top 5 moves:")
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (move, prob) in enumerate(sorted_moves, 1):
        visits = mcts.root.children[move].visit_count
        avg_value = mcts.root.children[move].value_sum / visits if visits > 0 else 0
        print(f"  {i}. {move} - Prob: {prob:.3f}, Visits: {visits}, Avg Value: {avg_value:.3f}")
    
    print(f"\n[BEST MOVE] {best_move} (visits: {mcts.root.children[best_move].visit_count})")
    
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Called when a new game begins. Use for initialization/cleanup."""
    print("Starting new game...")
    pass
