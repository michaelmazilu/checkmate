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
        return P, V
    else:
        # Hardcoded fallback (uniform distribution)
        P = {}
        V = 0.0
        return P, V


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Main inference function. Uses MCTS to select the best move.
    
    Time management:
    - Allocates time per move based on remaining time and estimated moves
    - Targets ~0.5-1 second per move to use time efficiently
    - Keeps safety margin to avoid timeouts
    """
    move_start_time = time.time()
    
    # Calculate time allocation per move
    # timeLeft is in milliseconds
    time_left_seconds = ctx.timeLeft / 1000.0
    
    # Estimate moves remaining (conservative estimate)
    # Games are typically 40-80 moves, so estimate conservatively
    # If we have lots of time, assume more moves left; if little time, assume fewer
    if time_left_seconds > 40:
        estimated_moves_remaining = 50  # Early game, many moves left
    elif time_left_seconds > 20:
        estimated_moves_remaining = 35  # Mid game
    else:
        estimated_moves_remaining = 20  # Late game, fewer moves left
    
    # Allocate time per move: use most of available time, but keep safety margin
    # Target: use ~80% of available time, leaving 20% safety margin
    # Per move: (time_left * 0.8) / estimated_moves
    time_per_move = (time_left_seconds * 0.8) / estimated_moves_remaining
    
    # Clamp between 0.3s (minimum) and 1.2s (maximum per move)
    # This ensures we don't go too fast or too slow
    time_per_move = max(0.3, min(1.2, time_per_move))
    
    # Add small buffer for overhead (neural net calls, etc.)
    search_time = time_per_move * 0.9  # Use 90% of allocated time for search
    
    # Initialize MCTS with time-based search
    mcts = MCTS(
        ctx.board, 
        neural_net, 
        num_simulations=1000,      # Fallback if time-based fails
        c_puct=1.0,                # Standard exploration
        dirichlet_alpha=0.3,       # Chess-specific noise parameter
        dirichlet_epsilon=0.0      # Disabled for speed
    )
    
    # Run time-based search
    mcts.search(max_time_seconds=search_time)
    
    elapsed = time.time() - move_start_time
    
    # Get best move and move probabilities
    best_move = mcts.get_best_move()
    move_probs = mcts.get_move_probabilities()
    
    if best_move is None:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Log the move probabilities for analysis (using Move objects as keys)
    ctx.logProbabilities(move_probs)
    
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Called when a new game begins. Use for initialization/cleanup."""
    print("Starting new game...")
    pass
