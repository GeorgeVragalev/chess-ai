import math
from Board import Board
from ChessPiece import *
from functools import wraps
from Logger import Logger, BoardRepr
import random

# Initializing a logger for logging game moves and states
logger = Logger()


# Decorator function to log the game tree during the minimax algorithm
def log_tree(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        board: Board = args[0]
        if board.log:
            depth = args[1]
            write_to_file(board, depth)
        return func(*args, **kwargs)
    return wrapper


# Function to write the board state to a file
def write_to_file(board: Board, current_depth):
    global logger
    if board.depth == current_depth:
        logger.clear()
    board_repr = BoardRepr(board.unicode_array_repr(), current_depth, board.evaluate())
    logger.append(board_repr)


# Check if a move is capturing an opponent's piece
def is_capturing_move(board, move):
    return isinstance(board[move[0]][move[1]], ChessPiece)


# The minimax function is the core of our AI's decision-making for chess moves.
# It calculates the best move by simulating potential moves up to a certain depth.
@log_tree
def minimax(board, depth, alpha, beta, max_player, save_move, data):
    # If we have reached the maximum depth or the game is over, return the board's score.
    if depth == 0 or board.is_terminal():
        data[1] = board.evaluate()
        return data

    # If it's the AI's turn (maximizing player).
    if max_player:
        # Set an initial lowest possible score.
        max_eval = -math.inf

        # Iterate over all squares on the chessboard.
        for i in range(8):
            for j in range(8):
                # If the square contains a chess piece and it's the opponent's piece.
                if isinstance(board[i][j], ChessPiece) and board[i][j].color != board.get_player_color():
                    # Get the piece.
                    piece = board[i][j]
                    # Get all possible moves for this piece.
                    moves = piece.filter_moves(piece.get_moves(board), board)

                    # Sort the moves so that capturing moves are considered first.
                    moves.sort(key=lambda sorted_move: is_capturing_move(board, sorted_move), reverse=True)

                    # Simulate each move.
                    for move in moves:
                        board.make_move(piece, move[0], move[1], keep_history=True)
                        # Recursively call the minimax function for the next depth and opposite player.
                        evaluation = minimax(board, depth - 1, alpha, beta, False, False, data)[1]
                        # If we need to save this move's data.
                        if save_move:
                            # Check if this move has a better evaluation than our current best.
                            if evaluation >= max_eval:
                                # If this move's evaluation is better than all previous moves.
                                if evaluation > data[1]:
                                    data.clear()
                                    data[1] = evaluation
                                    data[0] = [piece, move, evaluation]
                                # If this move's evaluation is the same as our best move's evaluation.
                                elif evaluation == data[1]:
                                    data[0].append([piece, move, evaluation])
                        # Undo the move to try the next one.
                        board.unmake_move(piece)
                        # Update our best evaluation if necessary.
                        max_eval = max(max_eval, evaluation)
                        # Alpha-beta pruning: Update the alpha value.
                        alpha = max(alpha, evaluation)
                        # If alpha is greater than or equal to beta, break out of the loop (pruning).
                        if beta <= alpha:
                            break
        return data

    # If it's the opponent's turn (minimizing player).
    else:
        # Set an initial highest possible score.
        min_eval = math.inf

        # Similar logic as the maximizing player but in the opposite direction.
        for i in range(8):
            for j in range(8):
                if isinstance(board[i][j], ChessPiece) and board[i][j].color == board.get_player_color():
                    piece = board[i][j]
                    moves = piece.get_moves(board)
                    moves.sort(key=lambda sorted_move: is_capturing_move(board, sorted_move), reverse=True)
                    for move in moves:
                        board.make_move(piece, move[0], move[1], keep_history=True)
                        evaluation = minimax(board, depth - 1, alpha, beta, True, False, data)[1]
                        board.unmake_move(piece)
                        min_eval = min(min_eval, evaluation)
                        beta = min(beta, evaluation)
                        if beta <= alpha:
                            break
        return data


# This function gradually deepens the depth the minimax function explores to find the best move.
def progressive_deepening(board, max_depth):
    best_evaluation = -math.inf
    best_move_data = []
    # Iterate through each depth until the max depth.
    for depth in range(1, max_depth + 1):
        # Get the best move's evaluation for this depth.
        current_evaluation = minimax(board, depth, -math.inf, math.inf, True, True, [[], 0])[1]
        # If this depth's move is better than our current best, update our best move data.
        if current_evaluation > best_evaluation:
            best_evaluation = current_evaluation
            best_move_data = minimax(board, depth, -math.inf, math.inf, True, True, [[], 0])
    return best_move_data


# Simple function to check if the game is still in the early phase (opening moves).
def is_in_opening(board):
    # If we have already made an opening move, return False.
    if board.opening_move_made:
        return False
    # Count the number of pieces captured.
    captured_count = 32 - len([piece for row in board for piece in row if isinstance(piece, ChessPiece)])
    # If fewer than 1 piece has been captured, consider it the opening phase.
    return captured_count < 1


# If we are in the opening phase, get a predefined opening move.
def get_opening_move():
    possible_moves = list(OPENING_MOVES.keys())
    if not possible_moves:
        return None
    # Randomly choose one of the predefined opening moves.
    start_square = random.choice(possible_moves)
    end_square = OPENING_MOVES[start_square]
    return start_square, end_square


# Function to decide which move the AI should make.
def get_ai_move(board):
    # Check if we should make an opening move.
    if is_in_opening(board):
        move = get_opening_move()
        if move:
            start_square, end_square = move[0], move[1]
            piece = board[start_square[0]][start_square[1]]
            board.make_move(piece, end_square[0], end_square[1])
            # Update that an opening move has been made.
            board.opening_move_made = True
            # If we are logging moves, log this move.
            if board.log:
                logger.write()
            return True

    # If not in the opening phase, use the progressive deepening function to decide the best move.
    moves_data = progressive_deepening(board, board.depth)
    if moves_data and len(moves_data[0]) > 0:
        best_score = max(moves_data[0], key=lambda x: x[2])[2]
        # Choose a move with the best score.
        piece_and_move = random.choice([move for move in moves_data[0] if move[2] == best_score])
        piece, move_coords = piece_and_move[0], piece_and_move[1]
        board.make_move(piece, move_coords[0], move_coords[1])
        if board.log:
            logger.write()
        return True
    return False


# A simple function for the AI to make a random move.
def get_random_move(board):
    pieces = []
    moves = []
    # Gather all of the AI's pieces and their moves.
    for i in range(8):
        for j in range(8):
            if isinstance(board[i][j], ChessPiece) and board[i][j].color != board.get_player_color():
                pieces.append(board[i][j])

    for piece in pieces[:]:
        piece_moves = piece.filter_moves(piece.get_moves(board), board)
        if len(piece_moves) == 0:
            pieces.remove(piece)
        else:
            moves.append(piece_moves)
    if len(pieces) == 0:
        return

    # Randomly choose a piece and make a random move with it.
    piece = random.choice(pieces)
    move = random.choice(moves[pieces.index(piece)])
    if isinstance(piece, ChessPiece) and len(move) > 0:
        board.make_move(piece, move[0], move[1])
