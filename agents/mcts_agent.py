import math
import time
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("mcts_agent")
class MctsAgent(Agent):
    """
    A Monte Carlo Tree Search agent using Upper Confidence Tree (UCT) for tree policy.
    """

    def __init__(self):
        super().__init__()
        self.name = "MctsAgent"
        self.simulation_time_limit = 2  # Time limit for the step method in seconds
        self.simulation_count = 5  # Number of rollouts per move
        self.c = math.sqrt(2)  # Exploration parameter for UCT

    def step(self, chess_board, player, opponent):
        start_time = time.time()
        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None  # No valid moves available

        sorted_moves = self.sort_moves(chess_board, valid_moves, player)

        # Initialize statistics for each move
        move_stats = {move: {"wins": 0, "visits": 0} for move in sorted_moves}

        # Perform simulations within the time limit
        for move in sorted_moves:
            if time.time() - start_time >= self.simulation_time_limit:
                break
            win_rate = self.simulate(chess_board, move, player, opponent, start_time)
            move_stats[move]["wins"] += win_rate * self.simulation_count
            move_stats[move]["visits"] += self.simulation_count

        # Select the move with the highest UCT value
        best_move = max(
            sorted_moves,
            key=lambda move: self.uct_score(
                move_stats[move]["wins"],
                move_stats[move]["visits"],
                sum(stat["visits"] for stat in move_stats.values()),
            ),
        )
        return best_move

    def simulate(self, board, move, player, opponent, start_time):
        """
        Perform Monte Carlo simulations for a given move.

        Args:
            board (list[list[int]]): The current game board.
            move (tuple): The move to simulate.
            player (int): The current player.
            opponent (int): The opponent player.

        Returns:
            float: The win rate for the move after simulations.
        """
        wins = 0
        for _ in range(self.simulation_count):
            if time.time() - start_time >= self.simulation_time_limit:
                break
            simulated_board = deepcopy(board)
            execute_move(simulated_board, move, player)

            # Perform a playout from this position
            winner = self.playout(simulated_board, opponent, player)
            if winner == player:
                wins += 1

        return wins / self.simulation_count if self.simulation_count > 0 else 0

    def playout(self, board, player, opponent):
        """
        Perform a random playout (default policy) from the current board state.
        """
        current_player = player
        while True:
            is_endgame, p0_score, p1_score = check_endgame(board, current_player, 3 - current_player)
            if is_endgame:
                break

            move = random_move(board, current_player)
            if move:
                execute_move(board, move, current_player)

            current_player = 3 - current_player  # Toggle between player and opponent

        # Determine the winner
        if p0_score > p1_score:
            return 1
        elif p1_score > p0_score:
            return 2
        return 0  # Draw

    def uct_score(self, wins, visits, total_visits):
        """
        Calculate the UCT score for a move.

        Args:
            wins (int): Total wins for the move.
            visits (int): Total visits for the move.
            total_visits (int): Total visits to the parent node.

        Returns:
            float: The UCT score.
        """
        if visits == 0:
            return float("inf")  # Prioritize unvisited moves
        exploitation = wins / visits
        exploration = self.c * math.sqrt(math.log(total_visits) / visits)
        return exploitation + exploration
    
    def evaluate_move(self, board, move, player):
        """
        Evaluate a move based on its impact on the board.
        
        Args:
            board (list[list[int]]): The current game board.
            move (tuple): The move to evaluate.
            player (int): The current player.

        Returns:
            int: A score representing the quality of the move.
        """
        temp_board = deepcopy(board)
        execute_move(temp_board, move, player)

        # Example heuristic: corner control and opponent's mobility
        corners = [(0, 0), (0, len(temp_board[0]) - 1), (len(temp_board) - 1, 0), (len(temp_board) - 1, len(temp_board[0]) - 1)]
        corner_score = sum(1 for corner in corners if temp_board[corner[0]][corner[1]] == player) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - player) * -25

        # Mobility: Number of valid moves for the opponent
        opponent = 3 - player
        opponent_moves = len(get_valid_moves(temp_board, opponent))
        mobility_penalty = -opponent_moves

        # Stable discs: discs that cannot be flipped
        stable_score = sum(1 for x in range(board.shape[0]) for y in range(board.shape[1])
                            if board[x, y] == player and self.is_stable(board, (x, y), player))

        # Combine heuristic components
        return corner_score + mobility_penalty + corner_penalty + stable_score

    def is_stable(self, board, position, color):
        """
        Check if a disc is stable (cannot be flipped).

        Parameters:
        - board: 2D numpy array representing the game board.
        - position: Tuple (x, y) of the disc's position.
        - color: Integer representing the disc's color.

        Returns:
        - bool: True if the disc is stable, False otherwise.
        """
        x, y = position
        rows, cols = board.shape
        return board[x, y] == color and (
            (x == 0 or x == rows - 1) and (y == 0 or y == cols - 1) or  # Corner discs
            all(board[i][y] == color for i in range(rows)) or  # Full vertical column
            all(board[x][j] == color for j in range(cols))  # Full horizontal row
        )

    def sort_moves(self, board, moves, player):
        """
        Sort the moves based on their evaluation score.

        Args:
            board (list[list[int]]): The current game board.
            moves (list[tuple]): List of valid moves.
            player (int): The current player.

        Returns:
            list[tuple]: Sorted list of moves based on evaluation.
        """
        return sorted(moves, key=lambda move: self.evaluate_move(board, move, player), reverse=True)