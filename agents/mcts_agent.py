# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves
import random


@register_agent("mcts_agent")
class MctsAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super().__init__()
        self.name = "MctsAgent"
        # Reduced number of simulations per move for autoplay optimization
        self.simulation_count = 2000

    def step(self, chess_board, player, opponent):
        best_move = None
        best_score = float('-inf')

        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None  # No valid moves available

        for move in valid_moves:
            # Simulate with heuristic
            win_rate = self.simulate(chess_board, move, player, opponent)
            heuristic_score = self.evaluate_board(chess_board, player)
            combined_score = win_rate + heuristic_score

            if combined_score > best_score:
                best_score = combined_score
                best_move = move

        return best_move

    def simulate(self, board, move, player, opponent):
        """
        Perform Monte Carlo simulations for a given move, considering heuristics.

        Args:
            board (numpy.ndarray): The current game board.
            move (tuple): The move to simulate.
            player (int): The current player.
            opponent (int): The opponent player.

        Returns:
            float: The win rate for the move after simulations.
        """
        wins = 0
        total_simulations = self.simulation_count

        for _ in range(total_simulations):
            simulated_board = deepcopy(board)
            execute_move(simulated_board, move, player)

            winner = self.playout_with_heuristics(simulated_board, player, opponent)
            if winner == player:
                wins += 1

        return wins / total_simulations if total_simulations > 0 else 0

    def playout_with_heuristics(self, board, player, opponent):
        """
        Simulate a heuristic-based playout from the current board state.

        Args:
            board (numpy.ndarray): The current game board.
            player (int): The current player.
            opponent (int): The opponent player.

        Returns:
            int: The winner (1 for Player 1, 2 for Player 2, or 0 for a draw).
        """
        current_player = player
        max_turns = 100  # Safety limit for autoplay
        turn_count = 0

        while not check_endgame(board, player, opponent):
            valid_moves = get_valid_moves(board, current_player)
            if valid_moves:
                # Prioritize moves with higher heuristic values
                best_move = max(valid_moves, key=lambda move: self.evaluate_board(deepcopy(board), current_player))
                execute_move(board, best_move, current_player)
            else:
                # No valid moves, pass turn
                pass

            current_player = 3 - current_player  # Switch player
            turn_count += 1

            if turn_count > max_turns:
                print("Playout exceeded max_turns, terminating.")
                break

        # Evaluate final scores
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)

        if player_score > opponent_score:
            return player
        elif opponent_score > player_score:
            return opponent
        return 0  # Draw

    def evaluate_board(self, board, player):
        """
        Evaluate the board state for the given player.

        Args:
            board (numpy.ndarray): The current game board.
            player (int): The current player.

        Returns:
            int: The heuristic score of the board.
        """
        # Corner positions are highly valuable
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(1 for corner in corners if board[corner] == player) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - player) * -10

        # Mobility: the number of moves available to the opponent
        opponent = 3 - player
        opponent_moves = len(get_valid_moves(board, opponent))
        mobility_score = -opponent_moves

        # Combine heuristic components
        return corner_score + mobility_score