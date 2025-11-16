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
    
    MCTS Configuration for handling unexpected/random moves:
    - num_simulations: More simulations = more robust to surprises
    - c_puct: Higher = more exploration (1.5 recommended for diverse opponents)
    - dirichlet_noise: Adds exploration at root (AlphaZero trick)
    """
    start_time = time.time()
    
    # Initialize MCTS optimized for speed (<1 second per move)
    # Reduced simulations for fast response time
    mcts = MCTS(
        ctx.board, 
        neural_net, 
        num_simulations=50,         # Reduced for speed (was 2000)
        c_puct=1.0,                 # Standard exploration
        dirichlet_alpha=0.3,        # Chess-specific noise parameter
        dirichlet_epsilon=0.0       # Disabled for speed (was 0.25)
    )
    mcts.search()
    
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        print(f"[WARNING] MCTS took {elapsed:.2f}s (target: <1s)")
    
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
