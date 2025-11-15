import random
import math


class MCTSNode:
    """Represents a node in the MCTS tree (a board state)."""
    
    def __init__(self, board, parent=None, action_move=None, policy_prior=1.0):
        self.board = board.copy()
        self.parent = parent
        self.action_move = action_move  # Move that led to this state
        self.children = {}  # Dict: move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.policy_prior = policy_prior  # P(a|s) from NN
        
    def ucb_score(self, c_puct=1.0):
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
    
    def select_best_child(self, c_puct=1.0):
        """Select child with highest UCB score for tree traversal."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))
    
    def get_unexplored_moves(self):
        """Get legal moves that haven't been explored yet."""
        legal_moves = list(self.board.generate_legal_moves())
        unexplored = [m for m in legal_moves if m not in self.children]
        return unexplored


class MCTS:
    """Monte Carlo Tree Search implementation."""
    
    def __init__(self, board, neural_net_fn, num_simulations=100, c_puct=1.0):
        """
        Initialize MCTS search.
        
        Args:
            board: chess.Board object representing current position
            neural_net_fn: Function that takes FEN string and returns (P, V)
            num_simulations: Number of MCTS simulations to run
            c_puct: Exploration constant for UCB
        """
        self.root = MCTSNode(board)
        self.neural_net_fn = neural_net_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def search(self):
        """Run MCTS search for the specified number of simulations."""
        for i in range(self.num_simulations):
            self._run_simulation()
    
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
        while not node.get_unexplored_moves() and node.children:
            node = node.select_best_child(self.c_puct)
            
            # Stop if we reach a terminal position
            if node.board.is_game_over():
                break
        
        # EXPANSION: Create new child node if game isn't over
        if not node.board.is_game_over():
            unexplored_moves = node.get_unexplored_moves()
            if unexplored_moves:
                move = random.choice(unexplored_moves)
                
                # Create new board state
                new_board = node.board.copy()
                new_board.push(move)
                
                # Create new node with policy prior from NN
                _, _ = self.neural_net_fn(new_board.fen())  # Get policy for this position
                policy_prior = 1.0 / len(list(new_board.generate_legal_moves()))
                
                child_node = MCTSNode(
                    new_board, 
                    parent=node, 
                    action_move=move,
                    policy_prior=policy_prior
                )
                node.children[move] = child_node
                node = child_node
        
        # EVALUATION: Get value estimate from NN
        if node.board.is_game_over():
            value = self._get_terminal_value(node.board)
        else:
            _, value = self.neural_net_fn(node.board.fen())
        
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

