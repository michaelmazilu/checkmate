"""
Inference module for the trained chess neural network.
Use this to load the model and make predictions.
"""

import torch
import torch.nn as nn
import chess
import numpy as np
from typing import Dict, Tuple


class ChessNet(nn.Module):
    """Neural network for chess position evaluation (same as training)."""
    
    def __init__(self, input_dim=773, hidden_dim=512, max_moves=4672):
        super(ChessNet, self).__init__()
        
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


def encode_board(fen: str) -> np.ndarray:
    """Encode a chess board from FEN to feature vector."""
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


def decode_move_index(idx: int) -> str:
    """Decode a move index back to UCI format."""
    # Handle promotions
    if idx >= 4096 * 3:
        promotion = 'q'
        idx -= 4096 * 3
    elif idx >= 4096 * 2:
        promotion = 'r'
        idx -= 4096 * 2
    elif idx >= 4096:
        promotion = 'b'
        idx -= 4096
    else:
        promotion = None
    
    from_square = idx // 64
    to_square = idx % 64
    
    move_uci = chess.square_name(from_square) + chess.square_name(to_square)
    if promotion:
        move_uci += promotion
    
    return move_uci


class ChessModelInference:
    """Wrapper class for model inference."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the inference model.
        
        Args:
            model_path: Path to the saved model checkpoint
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        self.model = ChessNet(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            max_moves=config['max_moves']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def predict(self, fen: str, board: chess.Board = None) -> Tuple[Dict[chess.Move, float], float]:
        """
        Get policy priors and value estimate for a position.
        
        Args:
            fen: FEN string of the position
            board: Optional chess.Board object (for legal move filtering)
        
        Returns:
            P: Dict mapping legal Move objects to probabilities
            V: Position value estimate (-1 to +1)
        """
        # Encode board
        board_encoding = encode_board(fen)
        board_tensor = torch.FloatTensor(board_encoding).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
        
        # Convert to probabilities
        policy_probs = torch.exp(policy_logits).squeeze(0).cpu().numpy()
        value_scalar = value.item()
        
        # Filter to legal moves only
        if board is None:
            board = chess.Board(fen)
        
        legal_moves = list(board.legal_moves)
        
        # Map legal moves to their probabilities
        P = {}
        for move in legal_moves:
            # Encode the move to get its index
            move_idx = self._encode_move(move)
            if move_idx < len(policy_probs):
                P[move] = policy_probs[move_idx]
        
        # Normalize probabilities over legal moves
        total_prob = sum(P.values())
        if total_prob > 0:
            P = {move: prob / total_prob for move, prob in P.items()}
        else:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / len(legal_moves)
            P = {move: uniform_prob for move in legal_moves}
        
        return P, value_scalar
    
    def _encode_move(self, move: chess.Move) -> int:
        """
        Encode a chess.Move to an index (must match training encoding).
        
        Encoding scheme:
        - Normal moves: from_square * 64 + to_square (0-4095)
        - Promotions: 4096 + from_file * 32 + to_file * 4 + promotion_type
          (ensures all indices < 4672)
        """
        from_sq = move.from_square
        to_sq = move.to_square
        base_idx = from_sq * 64 + to_sq
        
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
            
            return promotion_idx
        
        return base_idx


# Example usage
if __name__ == "__main__":
    # Load model
    model = ChessModelInference("checkmate_model.pt")
    
    # Test position
    board = chess.Board()
    P, V = model.predict(board.fen(), board)
    
    print(f"Position: {board.fen()}")
    print(f"Value estimate: {V:.3f}")
    print(f"\nTop 5 moves:")
    
    sorted_moves = sorted(P.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (move, prob) in enumerate(sorted_moves, 1):
        print(f"  {i}. {move.uci()} - {prob:.4f}")

