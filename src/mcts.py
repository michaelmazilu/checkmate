import random
import math
import numpy as np


class MCTSNode:
    """Represents a node in the MCTS tree (a board state)."""
    
    def __init__(self, board, parent=None, action_move=None, policy_prior=1.0, copy_board=True):
        self.board = board.copy(stack=False) if copy_board else board
        self.parent = parent
        self.action_move = action_move  # Move that led to this state
        self.children = {}  # Dict: move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.policy_prior = policy_prior  # P(a|s) from NN
        self.fen = self.board.fen()
        self.is_terminal = self.board.is_game_over()
        self._legal_moves = None
        self._unexpanded_moves = None
        
    def ucb_score(self, c_puct=0.8):
        """
        Upper Confidence Bound score balancing exploration vs exploitation.
        
        UCB = Q(s,a)/N(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Where:
        - Q(s,a) = value sum for this action
        - N(s,a) = visit count for this action
        - P(s,a) = policy prior from NN
        - N(s) = parent visit count
        - c_puct = exploration constant (higher = more exploration)
        """
        if self.visit_count == 0:
            return float('inf')  # Unexplored nodes are highest priority
        
        # Exploitation: average value of this move
        exploitation = self.value_sum / self.visit_count
        
        # Exploration: bonus for unexplored moves, scaled by policy prior
        exploration = (
            c_puct * 
            self.policy_prior * 
            math.sqrt(self.parent.visit_count) / 
            (1 + self.visit_count)
        )
        
        return exploitation + exploration
    
    def select_best_child(self, c_puct=1):
        """Select child with highest UCB score for tree traversal."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))
    
    def legal_moves(self):
        """Cache legal moves for the node to avoid recomputation."""
        if self._legal_moves is None:
            if self.is_terminal:
                self._legal_moves = []
            else:
                self._legal_moves = list(self.board.generate_legal_moves())
        return self._legal_moves

    def legal_move_count(self):
        """Convenience helper for len(legal_moves)."""
        return len(self.legal_moves())

    def get_unexplored_moves(self):
        """Get legal moves that haven't been explored yet."""
        if self.is_terminal:
            return []
        if self._unexpanded_moves is None:
            self._unexpanded_moves = self.legal_moves().copy()
        return self._unexpanded_moves

    def mark_move_explored(self, move):
        """Remove a move from the unexpanded list once a child is created."""
        if self._unexpanded_moves is None:
            return
        try:
            self._unexpanded_moves.remove(move)
        except ValueError:
            pass


class MCTS:
    """
    Monte Carlo Tree Search implementation with exploration enhancements.
    
    Key features for handling unexpected/random moves:
    - Dirichlet noise at root for exploration
    - Adjustable exploration constants
    - Robust to off-policy positions
    """
    
    def __init__(
        self, 
        board, 
        neural_net_fn, 
        num_simulations=400, 
        c_puct=0.8,  # Increased default for more exploration
        dirichlet_alpha=0.3,  # Controls exploration noise
        dirichlet_epsilon=0.25  # Mix 25% noise with NN priors at root
    ):
        """
        Initialize MCTS search.
        
        Args:
            board: chess.Board object representing current position
            neural_net_fn: Function that takes FEN string and returns (P, V)
            num_simulations: Number of MCTS simulations to run
            c_puct: Exploration constant for UCB (higher = more exploration)
            dirichlet_alpha: Alpha parameter for Dirichlet noise (0.3 for chess)
            dirichlet_epsilon: How much noise to mix at root (0.25 = 25% noise, 75% NN)
        """
        self.root = MCTSNode(board)
        self.neural_net_fn = neural_net_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root_expanded = False  # Track if root has been expanded
        self._nn_cache = {}  # Cache NN evaluations per FEN
        
    def search(self):
        """Run MCTS search for the specified number of simulations."""
        for i in range(self.num_simulations):
            self._run_simulation()
            
            # Apply Dirichlet noise to root after first expansion
            if not self.root_expanded and self.root.children:
                self._add_dirichlet_noise_to_root()
                self.root_expanded = True
    
    def _add_dirichlet_noise_to_root(self):
        """
        Add Dirichlet noise to root node policy priors for exploration.
        
        This is the AlphaZero trick for handling unexpected positions:
        - Mix NN policy with random noise at the root
        - Encourages exploration of all moves, even "weird" ones
        - Helps find refutations to unexpected opponent moves
        
        Formula: P'(a) = (1-ε)*P(a) + ε*η(a)
        where η ~ Dirichlet(α)
        """
        if not self.root.children:
            return
        
        moves = list(self.root.children.keys())
        num_moves = len(moves)
        
        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_moves)
        
        # Mix noise with existing priors
        for i, move in enumerate(moves):
            child = self.root.children[move]
            # Blend: 75% NN policy + 25% random exploration
            child.policy_prior = (
                (1 - self.dirichlet_epsilon) * child.policy_prior +
                self.dirichlet_epsilon * noise[i]
            )
    
    def _evaluate_fen(self, fen):
        """Memoize NN evaluations to avoid recomputing the same FEN."""
        cached = self._nn_cache.get(fen)
        if cached is None:
            cached = self.neural_net_fn(fen)
            self._nn_cache[fen] = cached
        return cached
    
    def _run_simulation(self):
        """
        Execute one MCTS simulation:
        1. Selection: Traverse tree using UCB until reaching a node with unexplored moves
        2. Expansion: Add a new child node by exploring an unexplored move
        3. Evaluation: Use NN to estimate value of the new position
        4. Backpropagation: Update visit counts and value sums up the tree
        """
        node = self.root
        
        # SELECTION: Traverse tree using UCB
        while True:
            unexplored_moves = node.get_unexplored_moves()
            if unexplored_moves or not node.children or node.is_terminal:
                break
            node = node.select_best_child(self.c_puct)
        
        # EXPANSION: Create new child node if game isn't over
        if not node.is_terminal and unexplored_moves:
            # Get policy priors from NN for the CURRENT position (before move)
            P, _ = self._evaluate_fen(node.fen)
            
            # Choose which unexplored move to expand, weighted by policy priors
            if P and len(P) > 0:
                # Get priors for unexplored moves
                move_priors = {}
                for m in unexplored_moves:
                    # Use NN prior if available, else small default value
                    move_priors[m] = P.get(m, 1e-8)
                
                # Normalize to get selection probabilities
                total_prior = sum(move_priors.values())
                if total_prior > 0:
                    weights = [move_priors[m] / total_prior for m in unexplored_moves]
                    move = random.choices(unexplored_moves, weights=weights, k=1)[0]
                    policy_prior = move_priors[move]
                else:
                    # Fallback to uniform if all priors are zero
                    move = random.choice(unexplored_moves)
                    policy_prior = 1.0 / max(node.legal_move_count(), 1)
            else:
                # Fallback to uniform if NN returns no policy
                move = random.choice(unexplored_moves)
                policy_prior = 1.0 / max(node.legal_move_count(), 1)
            
            # Create new board state
            new_board = node.board.copy(stack=False)
            new_board.push(move)
            
            # Create new node with policy prior from NN
            child_node = MCTSNode(
                new_board, 
                parent=node, 
                action_move=move,
                policy_prior=policy_prior,
                copy_board=False
            )
            node.children[move] = child_node
            node.mark_move_explored(move)
            node = child_node
        
        # EVALUATION: Get value estimate from NN
        if node.is_terminal:
            value = self._get_terminal_value(node.board)
        else:
            _, value = self._evaluate_fen(node.fen)
        
        # BACKPROPAGATION: Update statistics up the tree
        self._backpropagation(node, value)
    
    def _get_terminal_value(self, board):
        """
        Get the value of a terminal board position.
        Returns -1 to +1, from current player's perspective.
        """
        if board.is_checkmate():
            # Current player is checkmated = losing position
            return -1.0
        elif board.is_stalemate():
            # Stalemate = draw
            return 0.0
        elif board.is_fivefold_repetition() or board.is_seventyfive_moves():
            # Draw by repetition or 75-move rule
            return 0.0
        else:
            return 0.0
    
    def _backpropagation(self, node, value):
        """
        Update visit counts and value sums from node up to root.
        Alternates value sign because each level represents the opponent's perspective.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            
            # Flip value sign for parent (opponent's perspective)
            value = -value
            current = current.parent
    
    def get_best_move(self):
        """
        Return the move with the highest visit count.
        This is the best move according to the search.
        """
        if not self.root.children:
            return None
        
        best_move = max(
            self.root.children.items(),
            key=lambda x: x[1].visit_count
        )[0]
        return best_move
    
    def get_move_probabilities(self):
        """
        Return probability distribution over moves based on visit counts.
        Moves with more visits are higher probability.
        """
        total_visits = sum(child.visit_count for child in self.root.children.values())
        
        probabilities = {}
        for move, child in self.root.children.items():
            prob = child.visit_count / total_visits if total_visits > 0 else 0
            probabilities[move] = prob
        
        return probabilities
