import math
from Board import Board
from ChessPiece import *
from functools import wraps
from Logger import Logger, BoardRepr
import random


logger = Logger()


def log_tree(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        board: Board = args[0]
        if board.log:
            depth = args[1]
            write_to_file(board, depth)
        return func(*args, **kwargs)
    return wrapper


def write_to_file(board: Board, current_depth):
    global logger
    if board.depth == current_depth:
        logger.clear()
    board_repr = BoardRepr(board.unicode_array_repr(), current_depth, board.evaluate())
    logger.append(board_repr)


def is_capturing_move(board, move):
    return isinstance(board[move[0]][move[1]], ChessPiece)


@log_tree
def minimax(board, depth, alpha, beta, max_player, save_move, data):

    if depth == 0 or board.is_terminal():
        data[1] = board.evaluate()
        return data

    if max_player:
        max_eval = -math.inf
        for i in range(8):
            for j in range(8):
                if isinstance(board[i][j], ChessPiece) and board[i][j].color != board.get_player_color():
                    piece = board[i][j]
                    moves = piece.filter_moves(piece.get_moves(board), board)

                    moves.sort(key=lambda sorted_move: is_capturing_move(board, sorted_move), reverse=True)

                    for move in moves:
                        board.make_move(piece, move[0], move[1], keep_history=True)
                        evaluation = minimax(board, depth - 1, alpha, beta, False, False, data)[1]
                        if save_move:
                            if evaluation >= max_eval:
                                if evaluation > data[1]:
                                    data.clear()
                                    data[1] = evaluation
                                    data[0] = [piece, move, evaluation]
                                elif evaluation == data[1]:
                                    data[0].append([piece, move, evaluation])
                        board.unmake_move(piece)
                        max_eval = max(max_eval, evaluation)
                        alpha = max(alpha, evaluation)
                        if beta <= alpha:
                            break
        return data
    else:
        min_eval = math.inf
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


def progressive_deepening(board, max_depth):
    best_evaluation = -math.inf
    best_move_data = []
    for depth in range(1, max_depth + 1):
        current_evaluation = minimax(board, depth, -math.inf, math.inf, True, True, [[], 0])[1]
        if current_evaluation > best_evaluation:
            best_evaluation = current_evaluation
            best_move_data = minimax(board, depth, -math.inf, math.inf, True, True, [[], 0])
    return best_move_data


def is_in_opening(board):
    # Simple heuristic: if fewer than 1 pieces have been captured, we are still in the opening
    if board.opening_move_made:
        return False
    captured_count = 32 - len([piece for row in board for piece in row if isinstance(piece, ChessPiece)])
    return captured_count < 1


def get_opening_move():
    possible_moves = list(OPENING_MOVES.keys())
    if not possible_moves:
        return None
    start_square = random.choice(possible_moves)
    end_square = OPENING_MOVES[start_square]
    return start_square, end_square


def get_ai_move(board):
    if is_in_opening(board):
        move = get_opening_move()
        if move:
            start_square, end_square = move[0], move[1]
            piece = board[start_square[0]][start_square[1]]
            board.make_move(piece, end_square[0], end_square[1])
            board.opening_move_made = True
            return True

    # Using progressive deepening with a max depth
    moves_data = progressive_deepening(board, board.depth)
    if moves_data and len(moves_data[0]) > 0:
        best_score = max(moves_data[0], key=lambda x: x[2])[2]
        piece_and_move = random.choice([move for move in moves_data[0] if move[2] == best_score])
        piece, move_coords = piece_and_move[0], piece_and_move[1]
        board.make_move(piece, move_coords[0], move_coords[1])
        return True
    return False


# def get_ai_move(board):
#     moves = minimax(board, board.depth, -math.inf, math.inf, True, True, [[], 0])
#     if board.log:
#         logger.write()
#     # moves = [[pawn, move, move_score], [..], [..],[..], total_score]
#     if len(moves[0]) == 0:
#         return False
#     best_score = max(moves[0], key=lambda x: x[2])[2]
#     piece_and_move = random.choice([move for move in moves[0] if move[2] == best_score])
#     piece = piece_and_move[0]
#     move = piece_and_move[1]
#     if isinstance(piece, ChessPiece) and len(move) > 0 and isinstance(move, tuple):
#         board.make_move(piece, move[0], move[1])
#     return True




def get_random_move(board):
    pieces = []
    moves = []
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
    piece = random.choice(pieces)
    move = random.choice(moves[pieces.index(piece)])
    if isinstance(piece, ChessPiece) and len(move) > 0:
        board.make_move(piece, move[0], move[1])
