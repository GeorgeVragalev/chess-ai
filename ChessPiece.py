import operator
from itertools import product

from position_values import *


class ChessPiece:

    # history is used to keep data, so board.unmake_move() works properly.
    eaten_pieces_history = []
    has_moved_history = []
    position_history = []

    def __init__(self, color, x, y, unicode):
        self.moved = False
        self.color = color
        self.x = x
        self.y = y
        self.type = self.__class__.__name__
        self.unicode = unicode

    def piece_to_index(self):
        piece_map = {
            'Pawn': 0,
            'Knight': 1,
            'Bishop': 2,
            'Rook': 3,
            'Queen': 4,
            'King': 5
        }

        index = piece_map[self.type]
        if self.color == "Black":
            index += 6

        return index


    def filter_moves(self, moves, board):
        final_moves = moves[:]
        for move in moves:
            board.make_move(self, move[0], move[1], keep_history=True)
            if board.king_is_threatened(self.color, move):
                final_moves.remove(move)
            board.unmake_move(self)
        return final_moves

    def get_moves(self, board):
        pass

    def get_last_eaten(self):
        return self.eaten_pieces_history.pop()

    def set_last_eaten(self, piece):
        self.eaten_pieces_history.append(piece)

    def set_position(self, x, y, keep_history):
        if keep_history:
            self.position_history.append(self.x)
            self.position_history.append(self.y)
            self.has_moved_history.append(self.moved)
        self.x = x
        self.y = y
        self.moved = True

    def set_old_position(self):
        position_y = self.position_history.pop()
        position_x = self.position_history.pop()
        self.y = position_y
        self.x = position_x
        self.moved = self.has_moved_history.pop()

    def get_score(self, board):
        return 0

    def __repr__(self):
        return '{}: {}|{},{}'.format(self.type, self.color, self.x, self.y)


class Pawn(ChessPiece):

    def get_moves(self, board):
        moves = []
        if board.game_mode == 0 and self.color == 'white' or board.game_mode == 1 and self.color == 'black':
            direction = 1
        else:
            direction = -1
        x = self.x + direction
        if board.has_empty_block(x, self.y):
            moves.append((x, self.y))
            if self.moved is False and board.has_empty_block(x + direction, self.y):
                moves.append((x + direction, self.y))
        if board.is_valid_move(x, self.y - 1):
            if board.has_opponent(self, x, self.y - 1):
                moves.append((x, self.y - 1))
        if board.is_valid_move(self.x + direction, self.y + 1):
            if board.has_opponent(self, x, self.y + 1):
                moves.append((x, self.y + 1))
        return moves

    def get_score(self, board):
        x = self.x
        y = self.y
        if self.color == "black":
            x, y = 7 - x, 7 - y  # Mirror the coordinates

        material_value = 10
        positional_value = SCALE_FACTOR * PAWN_POSITIONAL_VALUES[x][y]
        return material_value + positional_value


class Knight(ChessPiece):

    def get_moves(self, board):
        moves = []
        add = operator.add
        sub = operator.sub
        op_list = [(add, sub), (sub, add), (add, add), (sub, sub)]
        nums = [(1, 2), (2, 1)]
        combinations = list(product(op_list, nums))
        for comb in combinations:
            x = comb[0][0](self.x, comb[1][0])
            y = comb[0][1](self.y, comb[1][1])
            if board.has_empty_block(x, y) or board.has_opponent(self, x, y):
                moves.append((x, y))
        return moves

    def get_score(self, board):
        x = self.x
        y = self.y
        if self.color == "black":
            x, y = 7 - x, 7 - y  # Mirror the coordinates

        material_value = 30
        positional_value = SCALE_FACTOR * KNIGHT_POSITIONAL_VALUES[x][y]
        return material_value + positional_value


class Bishop(ChessPiece):

    def get_moves(self, board):
        moves = []
        add = operator.add
        sub = operator.sub
        operators = [(add, add), (add, sub), (sub, add), (sub, sub)]
        for ops in operators:
            for i in range(1, 9):
                x = ops[0](self.x, i)
                y = ops[1](self.y, i)
                if not board.is_valid_move(x, y) or board.has_friend(self, x, y):
                    break
                if board.has_empty_block(x, y):
                    moves.append((x, y))
                if board.has_opponent(self, x, y):
                    moves.append((x, y))
                    break
        return moves

    def get_score(self, board):
        x = self.x
        y = self.y
        if self.color == "black":
            x, y = 7 - x, 7 - y  # Mirror the coordinates

        material_value = 30
        positional_value = SCALE_FACTOR * BISHOP_POSITIONAL_VALUES[x][y]
        return material_value + positional_value


class Rook(ChessPiece):

    def get_moves(self, board):
        moves = []
        moves += self.get_vertical_moves(board)
        moves += self.get_horizontal_moves(board)
        return moves

    def get_vertical_moves(self, board):
        moves = []
        for op in [operator.add, operator.sub]:
            for i in range(1, 9):
                x = op(self.x, i)
                if not board.is_valid_move(x, self.y) or board.has_friend(self, x, self.y):
                    break
                if board.has_empty_block(x, self.y):
                    moves.append((x, self.y))
                if board.has_opponent(self, x, self.y):
                    moves.append((x, self.y))
                    break
        return moves

    def get_horizontal_moves(self, board):
        moves = []
        for op in [operator.add, operator.sub]:
            for i in range(1, 9):
                y = op(self.y, i)
                if not board.is_valid_move(self.x, y) or board.has_friend(self, self.x, y):
                    break
                if board.has_empty_block(self.x, y):
                    moves.append((self.x, y))
                if board.has_opponent(self, self.x, y):
                    moves.append((self.x, y))
                    break
        return moves

    def get_score(self, board):
        x = self.x
        y = self.y
        if self.color == "black":
            x, y = 7 - x, 7 - y  # Mirror the coordinates

        material_value = 50
        positional_value = SCALE_FACTOR * ROOK_POSITIONAL_VALUES[x][y]
        return material_value + positional_value


class Queen(ChessPiece):

    def get_moves(self, board):
        moves = []
        rook = Rook(self.color, self.x, self.y, self.unicode)
        bishop = Bishop(self.color, self.x, self.y, self.unicode)
        rook_moves = rook.get_moves(board)
        bishop_moves = bishop.get_moves(board)
        if rook_moves:
            moves.extend(rook_moves)
        if bishop_moves:
            moves.extend(bishop_moves)
        return moves

    def get_score(self, board):
        x = self.x
        y = self.y
        if self.color == "black":
            x, y = 7 - x, 7 - y  # Mirror the coordinates

        material_value = 90
        positional_value = SCALE_FACTOR * QUEEN_POSITIONAL_VALUES[x][y]
        return material_value + positional_value


class King(ChessPiece):

    def get_moves(self, board):
        moves = []
        moves += self.get_horizontal_moves(board)
        moves += self.get_vertical_moves(board)
        return moves

    def get_vertical_moves(self, board):
        moves = []
        for op in [operator.add, operator.sub]:
            x = op(self.x, 1)
            if board.has_empty_block(x, self.y) or board.has_opponent(self, x, self.y):
                moves.append((x, self.y))
            if board.has_empty_block(x, self.y + 1) or board.has_opponent(self, x, self.y + 1):
                moves.append((x, self.y+1))
            if board.has_empty_block(x, self.y - 1) or board.has_opponent(self, x, self.y - 1):
                moves.append((x, self.y - 1))
        return moves

    def get_horizontal_moves(self, board):
        moves = []
        for op in [operator.add, operator.sub]:
            y = op(self.y, 1)
            if board.has_empty_block(self.x, y) or board.has_opponent(self, self.x, y):
                moves.append((self.x, y))
        return moves

    def get_king_safety_value(self, board):
        x = self.x
        y = self.y
        # Calculate the safety value based on the pieces surrounding the king
        adjacent_positions = set([
            (x + i, y + j)
            for i in [-1, 0, 1] for j in [-1, 0, 1]
            if 0 <= x + i < 8 and 0 <= y + j < 8 and not (i == 0 and j == 0)
            # Ensure valid board position and exclude the king's position
        ])

        friendlies_around = 0
        enemies_around = 0

        if self.color == "white":
            friendlies = board.whites
            enemies = board.blacks
        else:
            friendlies = board.blacks
            enemies = board.whites

        for piece in friendlies:
            if (piece.x, piece.y) in adjacent_positions:
                friendlies_around += 1

        for piece in enemies:
            if (piece.x, piece.y) in adjacent_positions:
                enemies_around += 1

        # Here, we're using simple illustrative values:
        # A bonus of 1 for each friendly piece and a penalty of 5 for each opponent piece.
        # These values can be fine-tuned.
        king_safety_value = friendlies_around * 0.5 - enemies_around * 1

        return king_safety_value

    def get_score(self, board):
        x = self.x
        y = self.y
        if self.color == "black":
            x, y = 7 - x, 7 - y  # Mirror the coordinates

        material_value = 1000
        positional_value = SCALE_FACTOR * KING_POSITIONAL_VALUES[x][y]


        # get king safety value
        king_safety = self.get_king_safety_value(board)

        return material_value + positional_value + king_safety
