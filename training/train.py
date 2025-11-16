import modal
import json
from pathlib import Path

# Create Modal app
app = modal.App("checkmate-training")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "chess",
        "tqdm",
        "huggingface_hub",
    )
)

# Create a volume to persist models and data
volume = modal.Volume.from_name("checkmate-models", create_if_missing=True)

# Configuration
BATCH_SIZE = 2048  # Increased significantly for speed (A10G has 24GB VRAM)
LEARNING_RATE = 0.0005  # Adjusted for larger batch: 0.001 / sqrt(4) = 0.0005
EPOCHS = 30  # Extended for Stockfish dataset training
HIDDEN_DIM = 512
MODEL_PATH = "/models/checkmate_model.pt"
CHECKPOINT_DIR = "/models/checkpoints"
HF_REPO_ID = "raf-fonseca/checkmate-chess"  # Your HuggingFace repo
USE_MIXED_PRECISION = True  # Enable FP16 training for 2-3x speed boost

# Memory optimization
USE_STREAMING = False  # Set to True to use disk streaming instead of loading all data into RAM
                       # Streaming: slower but uses minimal memory
                       # In-memory: faster but requires ~128GB RAM for 111M examples


# Create a separate volume for training data
data_volume = modal.Volume.from_name("checkmate-training-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",  # Upgraded to A10G GPU (24GB VRAM, faster than T4)
    memory=131072,  # 128GB RAM to handle 111M examples in memory
    timeout=86400,  # 24 hour timeout
    volumes={
        "/models": volume,
        "/data": data_volume,
    },
)
def train_model(data_path: str = None, resume_from_checkpoint: str = None):
    """
    Train the chess neural network.
    
    Args:
        data_path: Path to training data JSONL file (defaults to combined_elite.jsonl)
        resume_from_checkpoint: Path to checkpoint to resume from (e.g., "/models/checkpoints/checkpoint_epoch_5.pt")
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import chess
    import numpy as np
    from tqdm import tqdm
    
    print("=" * 80)
    print("CHECKMATE NEURAL NETWORK TRAINING")
    print("=" * 80)
    
    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
    
    class ChessNet(nn.Module):
        """
        Neural network for chess position evaluation.
        
        Inputs: Board position (FEN)
        Outputs: 
            - Policy (P): Probability distribution over all possible moves
            - Value (V): Position evaluation (-1 to +1)
        """
        
        def __init__(self, input_dim=773, hidden_dim=512, max_moves=4672):
            """
            Args:
                input_dim: Size of board representation (773 for our encoding)
                hidden_dim: Hidden layer dimension
                max_moves: Maximum possible chess moves (all from-to square combinations)
            """
            super(ChessNet, self).__init__()
            
            # Shared body
            self.body = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            )
            
            # Policy head (outputs probability distribution over moves)
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, max_moves),
                nn.LogSoftmax(dim=1)  # LogSoftmax for NLLLoss
            )
            
            # Value head (outputs position evaluation)
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Tanh()  # Output between -1 and 1
            )
        
        def forward(self, x):
            """
            Forward pass.
            
            Returns:
                policy_logits: Log probabilities over all moves [batch_size, max_moves]
                value: Position evaluation [batch_size, 1]
            """
            features = self.body(x)
            policy = self.policy_head(features)
            value = self.value_head(features)
            return policy, value
    
    # ========================================================================
    # DATA ENCODING
    # ========================================================================
    
    def encode_board(fen: str) -> np.ndarray:
        """
        Encode a chess board from FEN to a feature vector.
        
        Encoding (773 dimensions):
        - 768 dims: 12 piece types × 64 squares (one-hot)
        - 1 dim: Turn (0=white, 1=black)
        - 4 dims: Castling rights (KQkq)
        
        Returns:
            numpy array of shape (773,)
        """
        board = chess.Board(fen)
        encoding = np.zeros(773, dtype=np.float32)
        
        # Piece encoding (768 dims)
        piece_types = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # White pieces: 0-5, Black pieces: 6-11
                piece_idx = piece_types.index(piece.piece_type)
                if piece.color == chess.BLACK:
                    piece_idx += 6
                
                # One-hot encoding: piece_type * 64 + square
                encoding[piece_idx * 64 + square] = 1.0
        
        # Turn (768)
        encoding[768] = 1.0 if board.turn == chess.BLACK else 0.0
        
        # Castling rights (769-772)
        encoding[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        encoding[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        encoding[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        encoding[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        return encoding
    
    def encode_move(move_uci: str) -> int:
        """
        Encode a UCI move to an index.
        
        Encoding scheme (ensures all indices < 4672):
        - Normal moves: from_square * 64 + to_square (indices 0-4095)
        - Promotions: 4096 + (from_square % 8) * 32 + to_square % 8 * 4 + promotion_type
          where promotion_type: knight=0, bishop=1, rook=2, queen=3
          
        Promotions can only occur from ranks 2 or 7, so from_square % 8 gives file (0-7)
        and to_square % 8 also gives file (0-7).
        This gives 8 * 8 * 4 = 256 possible promotion moves (indices 4096-4351)
        
        Total max index: 4096 + 255 = 4351 < 4672 ✓
        
        Returns:
            integer index for the move (0 to 4351)
        """
        try:
            move = chess.Move.from_uci(move_uci)
        except Exception as e:
            print(f"WARNING: Invalid UCI move '{move_uci}': {e}, returning 0")
            return 0
        
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Validate square indices
        if not (0 <= from_sq < 64 and 0 <= to_sq < 64):
            print(f"WARNING: Invalid square indices for move '{move_uci}': from={from_sq}, to={to_sq}, returning 0")
            return 0
        
        # Base index for normal moves
        base_idx = from_sq * 64 + to_sq
        
        # Sanity check base index
        if base_idx >= 4096:
            print(f"WARNING: Base index {base_idx} >= 4096 for move '{move_uci}', capping at 4095")
            base_idx = 4095
        
        # Handle promotions
        if move.promotion:
            # Promotion encoding starts at 4096
            promotion_type = {
                chess.KNIGHT: 0,
                chess.BISHOP: 1,
                chess.ROOK: 2,
                chess.QUEEN: 3,
            }.get(move.promotion, 3)  # Default to queen
            
            # For promotions, encode as: file_from * 32 + file_to * 4 + promo_type
            # This gives max index: 7 * 32 + 7 * 4 + 3 = 224 + 28 + 3 = 255
            from_file = from_sq % 8
            to_file = to_sq % 8
            
            promotion_idx = 4096 + from_file * 32 + to_file * 4 + promotion_type
            
            # Final safety check
            if promotion_idx >= 4672:
                print(f"WARNING: Promotion index {promotion_idx} >= 4672 for move '{move_uci}', using base index")
                return base_idx if base_idx < 4672 else 0
            
            return promotion_idx
        
        return base_idx
    
    # ========================================================================
    # DATASET
    # ========================================================================
    
    class ChessDataset(Dataset):
        """Dataset for chess positions with moves and values - memory efficient version."""
        
        def __init__(self, data_path=None, sample_data=None, max_examples=None, use_streaming=False):
            """
            Args:
                data_path: Path to JSONL file
                sample_data: Pre-loaded data list
                max_examples: Maximum number of examples to load (for memory management)
                use_streaming: If True, stream data from disk instead of loading into memory
            """
            self.use_streaming = use_streaming
            self.data_path = data_path
            
            if use_streaming:
                # Streaming mode - only count lines, don't load into memory
                print(f"📊 Counting examples in {data_path}...")
                self.num_examples = 0
                with open(data_path, 'r') as f:
                    for i, _ in enumerate(f):
                        if max_examples and i >= max_examples:
                            break
                        self.num_examples += 1
                        if (i + 1) % 10_000_000 == 0:
                            print(f"  Counted {(i+1):,} examples...")
                print(f"✓ Using streaming mode with {self.num_examples:,} examples")
                print(f"💾 Memory efficient: Data will be loaded on-demand")
                
                # Build line offset index for random access
                print(f"🔍 Building line index for random access...")
                self.line_offsets = [0]
                with open(data_path, 'rb') as f:
                    count = 0
                    while f.readline():
                        if max_examples and count >= max_examples:
                            break
                        self.line_offsets.append(f.tell())
                        count += 1
                        if (count + 1) % 10_000_000 == 0:
                            print(f"  Indexed {(count+1):,} lines...")
                self.line_offsets = self.line_offsets[:-1]  # Remove last offset
                print(f"✓ Line index built")
                
            elif sample_data:
                # Use provided sample data (load into memory)
                self.data = sample_data
                self.use_streaming = False
            elif data_path:
                # Load from JSONL file with optional limit
                self.data = []
                print(f"Loading training data from {data_path}...")
                
                with open(data_path, 'r') as f:
                    for i, line in enumerate(f):
                        if max_examples and i >= max_examples:
                            print(f"Reached max_examples limit: {max_examples:,}")
                            break
                        self.data.append(json.loads(line))
                        
                        # Progress indicator every 10M examples
                        if (i + 1) % 10_000_000 == 0:
                            print(f"  Loaded {(i+1):,} examples...")
                
                print(f"✓ Loaded {len(self.data):,} training examples")
            else:
                self.data = []
                self.use_streaming = False
        
        def __len__(self):
            if self.use_streaming:
                return self.num_examples
            return len(self.data)
        
        def __getitem__(self, idx):
            if self.use_streaming:
                # Read single line from file on-demand
                with open(self.data_path, 'r') as f:
                    f.seek(self.line_offsets[idx])
                    line = f.readline()
                    item = json.loads(line)
            else:
                item = self.data[idx]
            
            # Encode board position
            board_encoding = encode_board(item['fen'])
            
            # Encode move
            move_idx = encode_move(item['move'])
            
            # Value (outcome)
            value = float(item['value'])
            
            return (
                torch.FloatTensor(board_encoding),
                torch.LongTensor([move_idx]),
                torch.FloatTensor([value])
            )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\n[1/5] Preparing data...")
    
    # Default data path (8.1M examples) - in Modal volume
    DEFAULT_DATA_PATH = "/data/combined_elite.jsonl"
    
    # Check if data file exists
    import os
    if not os.path.exists(DEFAULT_DATA_PATH) and not data_path:
        raise FileNotFoundError(
            f"Training data not found at {DEFAULT_DATA_PATH}\n"
            f"Please upload your data file first:\n"
            f"  modal volume put checkmate-training-data training/data/combined_elite.jsonl /combined_elite.jsonl"
        )
    
    # Use custom path if provided, otherwise use default
    if data_path:
        print(f"Loading training data from: {data_path}")
        dataset = ChessDataset(data_path=data_path, use_streaming=USE_STREAMING)
    else:
        print(f"Loading training data from: {DEFAULT_DATA_PATH}")
        dataset = ChessDataset(data_path=DEFAULT_DATA_PATH, use_streaming=USE_STREAMING)
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # Parallel data loading for speed
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    print(f"Created DataLoader with batch_size={BATCH_SIZE}")
    if USE_STREAMING:
        print(f"💾 Using streaming mode: minimal memory usage")
    else:
        print(f"⚡ Using in-memory mode: faster but high memory usage")
    
    print("\n[2/5] Initializing model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ChessNet(input_dim=773, hidden_dim=HIDDEN_DIM, max_moves=4672)
    model = model.to(device)
    
    # Loss functions
    policy_criterion = nn.NLLLoss()  # Negative log-likelihood for policy
    value_criterion = nn.MSELoss()   # Mean squared error for value
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Mixed precision training setup
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)
    if USE_MIXED_PRECISION:
        print("✓ Mixed precision training enabled (FP16)")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_from_checkpoint:
        import os
        if os.path.exists(resume_from_checkpoint):
            print(f"\n[RESUMING] Loading checkpoint from {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('loss', float('inf'))
            
            print(f"[RESUMING] ✓ Resumed from epoch {start_epoch}")
            print(f"[RESUMING] Previous loss: {checkpoint.get('loss', 'N/A')}")
            print(f"[RESUMING] Will train epochs {start_epoch + 1} to {EPOCHS}")
        else:
            print(f"\n[WARNING] Checkpoint not found at {resume_from_checkpoint}")
            print("[WARNING] Starting from scratch")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n[3/5] Training...")
    
    # Create checkpoint directory
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize avg_loss in case no training occurs
    avg_loss = best_loss
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        try:
            for batch_idx, (boards, moves, values) in enumerate(progress_bar):
                boards = boards.to(device, non_blocking=True)
                moves = moves.squeeze(1).to(device, non_blocking=True)
                values = values.to(device, non_blocking=True)
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                    # Forward pass
                    policy_logits, value_pred = model(boards)
                    
                    # CRITICAL: Validate move indices before loss computation
                    invalid_moves = moves >= 4672
                    if invalid_moves.any():
                        print(f"\n❌ INVALID MOVE INDICES DETECTED at epoch {epoch+1}, batch {batch_idx}")
                        print(f"Invalid moves: {moves[invalid_moves].tolist()}")
                        print(f"Max index: {moves.max().item()}, Min index: {moves.min().item()}")
                        # Skip this batch
                        continue
                    
                    # Calculate losses
                    policy_loss = policy_criterion(policy_logits, moves)
                    value_loss = value_criterion(value_pred, values)
                    
                    # Combined loss (you can adjust the weighting)
                    loss = policy_loss + value_loss
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"\n❌ NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
                    print(f"Policy loss: {policy_loss.item()}, Value loss: {value_loss.item()}")
                    raise ValueError("NaN loss detected")
                
                # Mixed precision backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # Track metrics
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'policy': f'{policy_loss.item():.4f}',
                    'value': f'{value_loss.item():.4f}'
                })
        except Exception as e:
            print(f"\n❌ Error during epoch {epoch+1}, batch {batch_idx}: {e}")
            # Save emergency checkpoint
            emergency_path = f"{CHECKPOINT_DIR}/emergency_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / max(batch_idx, 1),
                'config': {'input_dim': 773, 'hidden_dim': HIDDEN_DIM, 'max_moves': 4672}
            }, emergency_path)
            print(f"Emergency checkpoint saved to {emergency_path}")
            volume.commit()
            raise
        
        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        avg_policy_loss = policy_loss_sum / len(train_loader)
        avg_value_loss = value_loss_sum / len(train_loader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Avg Value Loss: {avg_value_loss:.4f}")
        
        # Save checkpoint after every epoch
        checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'config': {
                'input_dim': 773,
                'hidden_dim': HIDDEN_DIM,
                'max_moves': 4672,
            }
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save as best model if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'config': {
                    'input_dim': 773,
                    'hidden_dim': HIDDEN_DIM,
                    'max_moves': 4672,
                }
            }, MODEL_PATH)
            print(f"  ✓ New best model saved (loss: {avg_loss:.4f})")
    
    print("\n[4/5] Finalizing...")
    
    # Commit the volume to persist all checkpoints and best model
    volume.commit()
    print(f"Best model saved to {MODEL_PATH}")
    print(f"All {EPOCHS} checkpoints saved to {CHECKPOINT_DIR}")
    
    print("\n[5/5] Testing inference...")
    
    # Test inference on a sample position
    model.eval()
    with torch.no_grad():
        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        test_board = torch.FloatTensor(encode_board(test_fen)).unsqueeze(0).to(device)
        
        policy_logits, value = model(test_board)
        
        print(f"Test FEN: {test_fen}")
        print(f"Value prediction: {value.item():.4f}")
        print(f"Top 5 policy moves (indices): {torch.topk(policy_logits, 5).indices.cpu().numpy()}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best model: {MODEL_PATH}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {CHECKPOINT_DIR}/checkpoint_epoch_*.pt")
    
    return {
        'model_path': MODEL_PATH,
        'best_loss': best_loss,
        'final_epoch_loss': avg_loss,
        'epochs': EPOCHS
    }


@app.function(
    image=image,
    volumes={"/models": volume},
)
def download_model():
    """Download the trained model from Modal volume."""
    import base64
    
    with open(MODEL_PATH, 'rb') as f:
        model_bytes = f.read()
    
    return {
        'model_bytes': base64.b64encode(model_bytes).decode('utf-8'),
        'size': len(model_bytes)
    }


@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push_to_huggingface(repo_id: str):
    """
    Push the trained model to Hugging Face Hub.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "username/checkmate-chess")
    """
    from huggingface_hub import HfApi, create_repo
    import os
    
    print(f"Pushing model to Hugging Face: {repo_id}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            f"Please train the model first by running:\n"
            f"  modal run training/train.py\n"
            f"Then push with:\n"
            f"  modal run training/train.py --push-to-hf --hf-repo {repo_id}"
        )
    
    print(f"✓ Found model at {MODEL_PATH}")
    
    # Get HF token from environment (set via Modal secret)
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Please set up Modal secret: modal secret create huggingface-secret HF_TOKEN=your_token")
    
    # Initialize API
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=hf_token, exist_ok=True, repo_type="model")
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Create a model card
    model_card = """---
tags:
- chess
- reinforcement-learning
- mcts
- game-playing
license: mit
---

# Checkmate Chess Engine

A neural network trained to play chess using MCTS (Monte Carlo Tree Search) guidance.

## Model Description

This model evaluates chess positions and suggests moves. It outputs:
- **Policy (P)**: Probability distribution over legal moves
- **Value (V)**: Position evaluation from -1 (losing) to +1 (winning)

## Architecture

- Input: 773-dimensional board encoding (pieces, turn, castling rights)
- Hidden layers: 3x512 with ReLU + BatchNorm + Dropout
- Output heads:
  - Policy head: 4672-dim output (all possible moves)
  - Value head: Single scalar (-1 to +1)

## Training Data Format

The model is trained on game positions with format:
```json
{"fen": "...", "move": "e2e4", "value": -1}
```

## Usage

```python
from inference import ChessModelInference
import chess

# Load model
model = ChessModelInference("checkmate_model.pt")

# Get predictions
board = chess.Board()
P, V = model.predict(board.fen(), board)

print(f"Position value: {V}")
print(f"Best moves: {sorted(P.items(), key=lambda x: x[1], reverse=True)[:3]}")
```

## Integration with MCTS

This model is designed to work with MCTS for move selection. The policy priors guide
the search, while value estimates help evaluate unvisited positions.

## License

MIT License
"""
    
    # Save model card
    card_path = "/models/README.md"
    with open(card_path, 'w') as f:
        f.write(model_card)
    
    print("Uploading model file...")
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="checkmate_model.pt",
        repo_id=repo_id,
        token=hf_token,
        repo_type="model"
    )
    
    print("Uploading model card...")
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        token=hf_token,
        repo_type="model"
    )
    
    print(f"✅ Model successfully pushed to https://huggingface.co/{repo_id}")
    
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}"
    }


@app.local_entrypoint()
def main(
    data_path: str = None, 
    download: bool = False,
    push_to_hf: bool = False,
    hf_repo: str = None,
    resume_from: int = None,
    resume_from_checkpoint: str = None
):
    """
    Main entrypoint for training.
    
    Usage:
        # Train with default dataset (8.1M examples)
        modal run training/train.py
        
        # Resume from epoch 5 checkpoint
        modal run training/train.py --resume-from 5
        
        # Resume from a specific checkpoint file
        modal run training/train.py --resume-from-checkpoint /models/checkmate_model.pt
        
        # Train with custom data
        modal run training/train.py --data-path /data/stockfish_training.jsonl
        
        # Train with custom data AND resume from checkpoint
        modal run training/train.py --data-path /data/stockfish_training.jsonl --resume-from-checkpoint /models/checkmate_model.pt
        
        # Download trained model
        modal run training/train.py --download
        
        # Push to Hugging Face
        modal run training/train.py --push-to-hf --hf-repo username/checkmate-chess
    """
    if download:
        print("Downloading model...")
        result = download_model.remote()
        
        # Save locally
        import base64
        model_bytes = base64.b64decode(result['model_bytes'])
        
        output_path = Path(__file__).parent / "checkmate_model.pt"
        with open(output_path, 'wb') as f:
            f.write(model_bytes)
        
        print(f"Model downloaded to {output_path}")
        print(f"Size: {result['size'] / 1024 / 1024:.2f} MB")
    
    elif push_to_hf:
        if not hf_repo:
            print("ERROR: --hf-repo is required when using --push-to-hf")
            print("Example: modal run training/train.py --push-to-hf --hf-repo username/checkmate-chess")
            return
        
        print(f"Pushing model to Hugging Face: {hf_repo}")
        result = push_to_huggingface.remote(hf_repo)
        print(f"\n✅ Success! Model available at: {result['url']}")
    
    else:
        print("Starting training on Modal...")
        
        # Determine checkpoint path if resuming
        checkpoint_path = None
        if resume_from_checkpoint:
            # Direct checkpoint path provided
            checkpoint_path = resume_from_checkpoint
            print(f"Will resume from checkpoint: {checkpoint_path}")
        elif resume_from:
            # Epoch number provided
            checkpoint_path = f"/models/checkpoints/checkpoint_epoch_{resume_from}.pt"
            print(f"Will resume from epoch {resume_from} checkpoint: {checkpoint_path}")
        
        # Determine data source
        if data_path:
            print(f"Training with custom dataset: {data_path}")
        else:
            print(f"Training with default dataset: /data/combined_elite.jsonl")
        
        # Train the model
        try:
            result = train_model.remote(data_path=data_path, resume_from_checkpoint=checkpoint_path)
            print(f"\nTraining results: {result}")
            
            # Automatically push to HuggingFace after training
            print("\n" + "=" * 80)
            print("PUSHING TO HUGGING FACE")
            print("=" * 80)
            try:
                hf_result = push_to_huggingface.remote(HF_REPO_ID)
                print(f"\n✅ Success! Model available at: {hf_result['url']}")
                print("\nYour bot will automatically download this model on next run!")
            except Exception as e:
                print(f"\n⚠️ Failed to push to HuggingFace: {e}")
                print("You can manually push later with:")
                print(f"  modal run training/train.py --push-to-hf --hf-repo {HF_REPO_ID}")
        except Exception as e:
            print(f"\n❌ Training failed with error:")
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

