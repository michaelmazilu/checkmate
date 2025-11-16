"""
Improved training script with better hyperparameters for large-scale training.

Based on overfitting test results:
- Model architecture works perfectly (99% loss reduction)
- Issue: Hyperparameters not optimized for 111M examples

Key improvements:
1. Higher initial learning rate (0.003 vs 0.001)
2. Learning rate scheduling (cosine annealing)
3. Smaller batch size (256 vs 512) for better gradient estimates
4. Reduced dropout (0.2 vs 0.3) 
5. Gradient clipping to prevent instability
6. Warmup period for stable training start
"""

import modal
import json
from pathlib import Path

# Create Modal app
app = modal.App("checkmate-training-improved")

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

# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================
BATCH_SIZE = 256  # Reduced from 512 - smaller batches = better gradients
INITIAL_LEARNING_RATE = 0.003  # Increased from 0.001 - more aggressive learning
MIN_LEARNING_RATE = 0.0001  # Minimum LR for cosine annealing
EPOCHS = 30
HIDDEN_DIM = 512
DROPOUT = 0.2  # Reduced from 0.3 - less regularization
WARMUP_EPOCHS = 2  # Gradual LR warmup for stability
GRADIENT_CLIP = 1.0  # Clip gradients to prevent explosions

MODEL_PATH = "/models/checkmate_model.pt"
CHECKPOINT_DIR = "/models/checkpoints"
HF_REPO_ID = "raf-fonseca/checkmate-chess"

# Memory optimization
USE_STREAMING = False  # Set to True for disk streaming (slower but less memory)

# Create a separate volume for training data
data_volume = modal.Volume.from_name("checkmate-training-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",  # A10G GPU (24GB VRAM)
    memory=131072,  # 128GB RAM for 111M examples
    timeout=86400,  # 24 hour timeout
    volumes={
        "/models": volume,
        "/data": data_volume,
    },
)
def train_model(data_path: str = None, resume_from_checkpoint: str = None):
    """
    Train the chess neural network with improved hyperparameters.
    
    Args:
        data_path: Path to training data JSONL file
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import chess
    import numpy as np
    from tqdm import tqdm
    import math
    
    print("=" * 80)
    print("CHECKMATE NEURAL NETWORK TRAINING (IMPROVED)")
    print("=" * 80)
    print("\n🎯 Hyperparameter Improvements:")
    print(f"  • Batch Size: {BATCH_SIZE} (was 512)")
    print(f"  • Initial LR: {INITIAL_LEARNING_RATE} (was 0.001)")
    print(f"  • Dropout: {DROPOUT} (was 0.3)")
    print(f"  • Gradient Clipping: {GRADIENT_CLIP}")
    print(f"  • LR Schedule: Cosine annealing with warmup")
    print()
    
    # ========================================================================
    # MODEL ARCHITECTURE (reduced dropout)
    # ========================================================================
    
    class ChessNet(nn.Module):
        def __init__(self, input_dim=773, hidden_dim=512, max_moves=4672, dropout=0.2):
            super(ChessNet, self).__init__()
            
            self.body = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),  # Reduced dropout
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),  # Reduced dropout
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            )
            
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, max_moves),
                nn.LogSoftmax(dim=1)
            )
            
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Tanh()
            )
        
        def forward(self, x):
            features = self.body(x)
            policy = self.policy_head(features)
            value = self.value_head(features)
            return policy, value
    
    # ========================================================================
    # DATA ENCODING (same as before)
    # ========================================================================
    
    def encode_board(fen: str) -> np.ndarray:
        board = chess.Board(fen)
        encoding = np.zeros(773, dtype=np.float32)
        
        piece_types = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = piece_types.index(piece.piece_type)
                if piece.color == chess.BLACK:
                    piece_idx += 6
                encoding[piece_idx * 64 + square] = 1.0
        
        encoding[768] = 1.0 if board.turn == chess.BLACK else 0.0
        encoding[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        encoding[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        encoding[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        encoding[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        return encoding
    
    def encode_move(move_uci: str) -> int:
        try:
            move = chess.Move.from_uci(move_uci)
        except Exception as e:
            return 0
        
        from_sq = move.from_square
        to_sq = move.to_square
        
        if not (0 <= from_sq < 64 and 0 <= to_sq < 64):
            return 0
        
        base_idx = from_sq * 64 + to_sq
        if base_idx >= 4096:
            base_idx = 4095
        
        if move.promotion:
            promotion_type = {
                chess.KNIGHT: 0,
                chess.BISHOP: 1,
                chess.ROOK: 2,
                chess.QUEEN: 3,
            }.get(move.promotion, 3)
            
            from_file = from_sq % 8
            to_file = to_sq % 8
            promotion_idx = 4096 + from_file * 32 + to_file * 4 + promotion_type
            
            if promotion_idx >= 4672:
                return base_idx if base_idx < 4672 else 0
            return promotion_idx
        
        return base_idx
    
    # ========================================================================
    # DATASET
    # ========================================================================
    
    class ChessDataset(Dataset):
        def __init__(self, data_path=None, sample_data=None, max_examples=None, use_streaming=False):
            self.use_streaming = use_streaming
            self.data_path = data_path
            
            if use_streaming:
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
                
                print(f"🔍 Building line index...")
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
                self.line_offsets = self.line_offsets[:-1]
                print(f"✓ Line index built")
                
            elif sample_data:
                self.data = sample_data
                self.use_streaming = False
            elif data_path:
                self.data = []
                print(f"Loading training data from {data_path}...")
                
                with open(data_path, 'r') as f:
                    for i, line in enumerate(f):
                        if max_examples and i >= max_examples:
                            break
                        self.data.append(json.loads(line))
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
                with open(self.data_path, 'r') as f:
                    f.seek(self.line_offsets[idx])
                    line = f.readline()
                    item = json.loads(line)
            else:
                item = self.data[idx]
            
            board_encoding = encode_board(item['fen'])
            move_idx = encode_move(item['move'])
            value = float(item['value'])
            
            return (
                torch.FloatTensor(board_encoding),
                torch.LongTensor([move_idx]),
                torch.FloatTensor([value])
            )
    
    # ========================================================================
    # LEARNING RATE SCHEDULE
    # ========================================================================
    
    def get_lr(epoch, batch_idx, batches_per_epoch):
        """
        Cosine annealing with warmup.
        
        - Warmup: Linear increase from 0 to INITIAL_LR over WARMUP_EPOCHS
        - Cosine: Decrease from INITIAL_LR to MIN_LR following cosine curve
        """
        total_steps = EPOCHS * batches_per_epoch
        warmup_steps = WARMUP_EPOCHS * batches_per_epoch
        current_step = epoch * batches_per_epoch + batch_idx
        
        if current_step < warmup_steps:
            # Warmup phase
            return INITIAL_LEARNING_RATE * current_step / warmup_steps
        else:
            # Cosine annealing phase
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return MIN_LEARNING_RATE + (INITIAL_LEARNING_RATE - MIN_LEARNING_RATE) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\n[1/5] Preparing data...")
    
    DEFAULT_DATA_PATH = "/data/combined_elite.jsonl"
    
    import os
    if not os.path.exists(DEFAULT_DATA_PATH) and not data_path:
        raise FileNotFoundError(f"Training data not found at {DEFAULT_DATA_PATH}")
    
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
        num_workers=0
    )
    
    print(f"Created DataLoader with batch_size={BATCH_SIZE}")
    if USE_STREAMING:
        print(f"💾 Using streaming mode: minimal memory usage")
    else:
        print(f"⚡ Using in-memory mode: faster training")
    
    print("\n[2/5] Initializing model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ChessNet(input_dim=773, hidden_dim=HIDDEN_DIM, max_moves=4672, dropout=DROPOUT)
    model = model.to(device)
    
    policy_criterion = nn.NLLLoss()
    value_criterion = nn.MSELoss()
    
    # Optimizer (will update LR manually)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_from_checkpoint:
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
    
    print("\n[3/5] Training with improved hyperparameters...")
    print(f"📈 Learning rate schedule: {INITIAL_LEARNING_RATE} → {MIN_LEARNING_RATE}")
    print(f"🔥 Warmup: {WARMUP_EPOCHS} epochs")
    print(f"✂️  Gradient clipping: {GRADIENT_CLIP}")
    print()
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    avg_loss = best_loss
    batches_per_epoch = len(train_loader)
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        try:
            for batch_idx, (boards, moves, values) in enumerate(progress_bar):
                # Update learning rate
                current_lr = get_lr(epoch, batch_idx, batches_per_epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                boards = boards.to(device)
                moves = moves.squeeze(1).to(device)
                values = values.to(device)
                
                # Forward pass
                policy_logits, value_pred = model(boards)
                
                # Validate move indices
                invalid_moves = moves >= 4672
                if invalid_moves.any():
                    print(f"\n❌ INVALID MOVE INDICES at batch {batch_idx}")
                    continue
                
                # Calculate losses
                policy_loss = policy_criterion(policy_logits, moves)
                value_loss = value_criterion(value_pred, values)
                loss = policy_loss + value_loss
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"\n❌ NaN loss at epoch {epoch+1}, batch {batch_idx}")
                    raise ValueError("NaN loss detected")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'policy': f'{policy_loss.item():.4f}',
                    'value': f'{value_loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
        except Exception as e:
            print(f"\n❌ Error during epoch {epoch+1}, batch {batch_idx}: {e}")
            emergency_path = f"{CHECKPOINT_DIR}/emergency_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / max(batch_idx, 1),
                'config': {'input_dim': 773, 'hidden_dim': HIDDEN_DIM, 'max_moves': 4672, 'dropout': DROPOUT}
            }, emergency_path)
            print(f"Emergency checkpoint saved to {emergency_path}")
            volume.commit()
            raise
        
        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        avg_policy_loss = policy_loss_sum / len(train_loader)
        avg_value_loss = value_loss_sum / len(train_loader)
        final_lr = get_lr(epoch, batches_per_epoch - 1, batches_per_epoch)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Avg Value Loss: {avg_value_loss:.4f}")
        print(f"  Final LR: {final_lr:.6f}")
        
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
                'dropout': DROPOUT,
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
                    'dropout': DROPOUT,
                }
            }, MODEL_PATH)
            print(f"  ✓ New best model saved (loss: {avg_loss:.4f})")
    
    print("\n[4/5] Finalizing...")
    volume.commit()
    print(f"Best model saved to {MODEL_PATH}")
    print(f"All {EPOCHS} checkpoints saved to {CHECKPOINT_DIR}")
    
    print("\n[5/5] Testing inference...")
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
    
    return {
        'model_path': MODEL_PATH,
        'best_loss': best_loss,
        'final_epoch_loss': avg_loss,
        'epochs': EPOCHS
    }


@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push_to_huggingface(repo_id: str):
    """Push the trained model to Hugging Face Hub."""
    from huggingface_hub import HfApi, create_repo
    import os
    
    print(f"Pushing model to Hugging Face: {repo_id}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found")
    
    api = HfApi()
    
    try:
        create_repo(repo_id, token=hf_token, exist_ok=True, repo_type="model")
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")
    
    print("Uploading model...")
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="checkmate_model.pt",
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
    push_to_hf: bool = False,
    hf_repo: str = None,
    resume_from: int = None,
    resume_from_checkpoint: str = None
):
    """
    Main entrypoint for improved training.
    
    Usage:
        # Start fresh training with improved hyperparameters
        modal run training/train_improved.py --data-path /data/stockfish_training.jsonl
        
        # Resume from epoch 20 (will use improved hyperparameters)
        modal run training/train_improved.py --data-path /data/stockfish_training.jsonl --resume-from 20
        
        # Push to Hugging Face
        modal run training/train_improved.py --push-to-hf --hf-repo username/checkmate-chess
    """
    if push_to_hf:
        if not hf_repo:
            print("ERROR: --hf-repo is required")
            return
        
        result = push_to_huggingface.remote(hf_repo)
        print(f"\n✅ Success! Model available at: {result['url']}")
    
    else:
        print("Starting IMPROVED training on Modal...")
        
        # Determine checkpoint path
        checkpoint_path = None
        if resume_from_checkpoint:
            checkpoint_path = resume_from_checkpoint
        elif resume_from:
            checkpoint_path = f"/models/checkpoints/checkpoint_epoch_{resume_from}.pt"
        
        # Train
        try:
            result = train_model.remote(data_path=data_path, resume_from_checkpoint=checkpoint_path)
            print(f"\nTraining results: {result}")
            
            # Auto-push to HuggingFace
            print("\n" + "=" * 80)
            print("PUSHING TO HUGGING FACE")
            print("=" * 80)
            try:
                hf_result = push_to_huggingface.remote(HF_REPO_ID)
                print(f"\n✅ Success! Model available at: {hf_result['url']}")
            except Exception as e:
                print(f"\n⚠️ Failed to push to HuggingFace: {e}")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

