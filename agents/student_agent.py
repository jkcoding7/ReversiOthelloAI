from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, execute_move, check_endgame
import copy
import time

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A Reversi agent that uses minimax with alpha-beta pruning.
    """

    def __init__(self):
        super().__init__()
        self.name = "StudentAgent"
        self.depth = 3  # Depth limit for the minimax search
        self.transposition_table = {}  # Dictionary for memoization

    def step(self, board, color, opponent):
        """
        Choose the best move using minimax with alpha-beta pruning and a time limit.
        """
        legal_moves = get_valid_moves(board, color)

        if not legal_moves:
            return None  # No valid moves available, pass turn

        ordered_moves = self.order_moves(legal_moves, board, color)

        # Start the timer
        start_time = time.time()
        time_limit = 2  # Time limit in seconds

        best_score = float('-inf')
        best_move = ordered_moves[0]
        alpha = float('-inf')
        beta = float('inf')

        for move in ordered_moves:
            if time.time() - start_time >= time_limit:
                break

            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)

            # Pass start_time and time_limit to minimax
            score = self.minimax(simulated_board, self.depth, False, alpha, beta, 3 - color, start_time, time_limit)
            
            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

        return best_move

    def minimax(self, board, depth, is_maximizing, alpha, beta, color, start_time, time_limit):
        """
        Minimax algorithm with alpha-beta pruning and time limit.
        """
        # Check if time limit is exceeded
        if time.time() - start_time >= time_limit:
            return alpha if is_maximizing else beta

        board_tuple = self.board_to_tuple(board)
        
        # Check if this board configuration is already in the transposition table
        if board_tuple in self.transposition_table:
            return self.transposition_table[board_tuple]

        legal_moves = get_valid_moves(board, color)

        if depth == 0 or not legal_moves:
            return self.evaluate_board(board, color)

        ordered_moves = self.order_moves(legal_moves, board, color)

        if is_maximizing:
            for move in ordered_moves:
                # Check if time limit is exceeded
                if time.time() - start_time >= time_limit:
                    break
                simulated_board = copy.deepcopy(board)
                execute_move(simulated_board, move, color)
                eval = self.minimax(simulated_board, depth - 1, False, alpha, beta, 3 - color, start_time, time_limit)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    return beta
            return alpha
        else:
            for move in ordered_moves:
                # Check if time limit is exceeded
                if time.time() - start_time >= time_limit:
                    break
                simulated_board = copy.deepcopy(board)
                execute_move(simulated_board, move, color)
                eval = self.minimax(simulated_board, depth - 1, True, alpha, beta, 3 - color, start_time, time_limit)
                beta = min(beta, eval)
                if beta <= alpha:
                    return alpha
            return beta
        
    def order_moves(self, legal_moves, board, color):
        """
        Order the legal moves based on a heuristic evaluation.

        Parameters:
        - legal_moves: List of valid moves.
        - board: Current game board.
        - color: The current player's color.

        Returns:
        - List of moves ordered by heuristic evaluation.
        """
        # Generate a score for each move based on the current board state
        move_scores = []
        for move in legal_moves:
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            score = self.evaluate_board(simulated_board, color)  # Heuristic evaluation of the move
            move_scores.append((score, move))
        
        # Sort moves by their heuristic score (higher is better for maximizing player)
        ordered_moves = [move for _, move in sorted(move_scores, reverse=True, key=lambda x: x[0])]
        
        return ordered_moves

    def board_to_tuple(self, board):
        """
        Convert the board to a tuple representation that can be used as a key for memoization.

        Parameters:
        - board: 2D numpy array representing the game board.

        Returns:
        - tuple: A tuple representation of the board.
        """
        return tuple(tuple(row) for row in board)

    def evaluate_board(self, board, color):
        """
        Evaluate the board state using a dynamic heuristic.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 or 2).

        Returns:
        - int: The evaluated score of the board.
        """
        # Corner positions are highly valuable
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(1 for corner in corners if board[corner] == color) * 25
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - color) * -25

        # Mobility: the number of moves the opponent can make
        opponent_moves = len(get_valid_moves(board, 3 - color))
        mobility_score = -opponent_moves

        # Stable discs: discs that cannot be flipped
        stable_score = sum(1 for x in range(board.shape[0]) for y in range(board.shape[1])
                            if board[x, y] == color and self.is_stable(board, (x, y), color))

        # Combine scores
        return corner_score + corner_penalty + mobility_score + stable_score

    def is_stable(self, board, position, color):
        """
        Check if a disc is stable (cannot be flipped), considering corners, rows, columns, and diagonals.

        Parameters:
        - board: 2D numpy array representing the game board.
        - position: Tuple (x, y) of the disc's position.
        - color: Integer representing the disc's color.

        Returns:
        - bool: True if the disc is stable, False otherwise.
        """
        x, y = position
        rows, cols = board.shape
        if board[x, y] != color:
            return False

        # Check if the disc is in one of the corners
        if (x == 0 or x == rows - 1) and (y == 0 or y == cols - 1):
            return True

        # Check full vertical column
        if all(board[i][y] == color for i in range(rows)):
            return True

        # Check full horizontal row
        if all(board[x][j] == color for j in range(cols)):
            return True

        # Check diagonal top-left to bottom-right
        if all(board[i][i] == color for i in range(min(rows, cols))):
            return True

        # Check diagonal top-right to bottom-left
        if all(board[i][cols - 1 - i] == color for i in range(min(rows, cols))):
            return True

        return False