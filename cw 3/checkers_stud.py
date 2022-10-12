#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rafał Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu Wstęp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji. 
Kod nie jest wzorem dobrej jakości programowania w Pythonie, nie jest również wzorem programowania obiektowego, może zawierać błędy.
Mam świadomość wielu jego braków ale nie mam czasu na jego poprawianie.

Zasady gry: https://en.wikipedia.org/wiki/English_draughts (w skrócie: wszyscy ruszają się po 1 polu. Pionki tylko w kierunku wroga, damki w dowolnym)
  z następującymi modyfikacjami: a) bicie nie jest wymagane,  b) dozwolone jest tylko pojedyncze bicie (bez serii).

Nalezy napisac funkcje minimax_a_b_recurr, minimax_a_b (woła funkcję rekurencyjną) i  evaluate, która ocenia stan gry

Chętni mogą ulepszać mój kod (trzeba oznaczyć komentarzem co zostało zmienione), mogą również dodać obsługę bicia wielokrotnego i wymagania bicia. Mogą również wdrożyć reguły: https://en.wikipedia.org/wiki/Russian_draughts
"""
import concurrent.futures
import math

import numpy as np
import pygame
from copy import deepcopy

FPS = 20

MINIMAX_DEPTH = 5

WIN_WIDTH = 800
WIN_HEIGHT = 800

BOARD_WIDTH = 8

FIELD_SIZE = WIN_WIDTH / BOARD_WIDTH
PIECE_SIZE = FIELD_SIZE / 2 - 8
MARK_THICK = 2
POS_MOVE_MARK_SIZE = PIECE_SIZE / 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Move:
    def __init__(self, piece, dest_row, dest_col, captures=None):
        self.piece = piece
        self.dest_row = dest_row
        self.dest_col = dest_col
        self.captures = captures

    def __str__(self) -> str:
        return f"{self.piece} - {self.dest_col} {self.dest_row} {self.captures}"


class Field:
    def draw(self):
        pass

    def is_empty(self):
        return True

    def is_white(self):
        return False

    def is_blue(self):
        return False

    def toogle_mark(self):
        pass

    def is_move_mark(self):
        return False

    def is_marked(self):
        return False

    def __str__(self):
        return "."


class PosMoveField(Field):
    def __init__(self, is_white, window, row, col, board, row_from, col_from, pos_move):
        self.__is_white = is_white
        self.__is_marked = False
        self.window = window
        self.row = row
        self.col = col
        self.board = board
        self.row_from = row_from
        self.col_from = col_from
        self.pos_move = pos_move

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def draw(self):
        x = self.col * FIELD_SIZE
        y = self.row * FIELD_SIZE
        pygame.draw.circle(self.window, RED, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2), POS_MOVE_MARK_SIZE)

    def is_empty(self):
        return True

    def is_move_mark(self):
        return True


class Pawn(Field):
    def __init__(self, is_white, window, row, col, board):
        self.__is_white = is_white
        self.__is_marked = False
        self.window = window
        self.row = row
        self.col = col
        self.board = board

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def __str__(self):
        if self.is_white():
            return "w"
        return "b"

    def is_king(self):
        return False

    def is_empty(self):
        return False

    def is_white(self):
        return self.__is_white

    def is_blue(self):
        return not self.__is_white

    def is_marked(self):
        return self.__is_marked

    def toogle_mark(self):
        if self.__is_marked:
            for pos_move in self.pos_moves:  # remove possible moves
                row = pos_move.dest_row
                col = pos_move.dest_col
                self.board.board[row][col] = Field()
            self.pos_moves = []
        else:  # self.is_marked
            self.pos_moves = self.board.get_piece_moves(self)
            for pos_move in self.pos_moves:
                row = pos_move.dest_row
                col = pos_move.dest_col
                self.board.board[row][col] = PosMoveField(False, self.window, row, col, self.board, self.row, self.col,
                                                          pos_move)

        self.__is_marked = not self.__is_marked

    def draw(self):
        if self.__is_white:
            cur_col = WHITE
        else:
            cur_col = BLUE
        x = self.col * FIELD_SIZE
        y = self.row * FIELD_SIZE
        pygame.draw.circle(self.window, cur_col, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2), PIECE_SIZE)

        if self.__is_marked:
            pygame.draw.circle(self.window, RED, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2), PIECE_SIZE + MARK_THICK,
                               MARK_THICK)


class King(Pawn):
    def __init__(self, pawn):
        super().__init__(pawn.is_white(), pawn.window, pawn.row, pawn.col, pawn.board)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def is_king(self):
        return True

    def __str__(self):
        if self.is_white():
            return "W"
        return "B"

    def draw(self):
        if self.is_white():
            cur_col = WHITE
        else:
            cur_col = BLUE
        x = self.col * FIELD_SIZE
        y = self.row * FIELD_SIZE
        pygame.draw.circle(self.window, cur_col, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2), PIECE_SIZE)
        pygame.draw.circle(self.window, GREEN, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2), PIECE_SIZE / 2)

        if self.is_marked():
            pygame.draw.circle(self.window, RED, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2), PIECE_SIZE + MARK_THICK,
                               MARK_THICK)


class Board:
    def __init__(self, window):  # row, col
        self.board = []  # np.full((BOARD_WIDTH, BOARD_WIDTH), None)
        self.window = window
        self.marked_piece = None
        self.something_is_marked = False
        self.white_turn = True
        self.white_fig_left = 12
        self.blue_fig_left = 12

        self.__set_pieces()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        result.board = deepcopy(self.board)
        return result

    def __str__(self):
        to_ret = ""
        for row in range(8):
            for col in range(8):
                to_ret += str(self.board[row][col])
            to_ret += "\n"
        return to_ret

    def __set_pieces(self):
        for row in range(8):
            self.board.append([])
            for col in range(8):
                self.board[row].append(Field())

        for row in range(3):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(False, self.window, row, col, self)

        for row in range(5, 8):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(True, self.window, row, col, self)

    def get_piece_moves(self, piece):
        pos_moves = []
        row = piece.row
        col = piece.col
        if piece.is_blue():
            enemy_is_white = True
        else:
            enemy_is_white = False

        if piece.is_white() or (piece.is_blue() and piece.is_king()):
            dir_y = -1
            if row > 0:
                new_row = row + dir_y
                if col > 0:
                    new_col = col - 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # ruch zwiazany z biciem
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y >= 0 and new_col - 1 >= 0 and \
                            self.board[new_row + dir_y][new_col - 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col - 1, self.board[new_row][new_col]))

                if col < BOARD_WIDTH - 1:
                    new_col = col + 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # ruch zwiazany z biciem
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y >= 0 and new_col + 1 < BOARD_WIDTH and \
                            self.board[new_row + dir_y][new_col + 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col + 1, self.board[new_row][new_col]))

        if piece.is_blue() or (piece.is_white() and self.board[row][col].is_king()):
            dir_y = 1
            if row < BOARD_WIDTH - 1:
                new_row = row + dir_y
                if col > 0:
                    new_col = col - 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y < BOARD_WIDTH and new_col - 1 >= 0 and \
                            self.board[new_row + dir_y][new_col - 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col - 1, self.board[new_row][new_col]))

                if col < BOARD_WIDTH - 1:
                    new_col = col + 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # ruch zwiazany z biciem
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y < BOARD_WIDTH and new_col + 1 < BOARD_WIDTH and \
                            self.board[new_row + dir_y][new_col + 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col + 1, self.board[new_row][new_col]))
        return pos_moves

    # ToDo
    def evaluate(self):  # white is enemy
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if self.board[row][col].is_white():
                        if self.board[row][col].is_king():
                            h -= 10
                        else:
                            h -= 1
                    else:
                        if self.board[row][col].is_king():
                            h += 10
                        else:
                            h += 1
        return h

    def evaluate_1(self):  # white is enemy
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if self.board[row][col].is_white():
                        if self.board[row][col].is_king():
                            h -= 10
                        else:
                            h -= 1
                    else:
                        if self.board[row][col].is_king():
                            h += 10
                        else:
                            h += 1
        return h + self._integrity(True)

    def evaluate_2(self):  # white is enemy
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if self.board[row][col].is_white():
                        if self.board[row][col].is_king():
                            h -= 10
                        else:
                            h -= 1
                    else:
                        if self.board[row][col].is_king():
                            h += 10
                        else:
                            if row >= BOARD_WIDTH / 2:
                                h += 7
                            else:
                                h += 5
        return h

    def evaluate_3(self):  # white is enemy
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if self.board[row][col].is_white():
                        if self.board[row][col].is_king():
                            h -= 10
                        else:
                            h -= 1
                    else:
                        if self.board[row][col].is_king():
                            h += 10
                        else:
                            h += 5 + row + 1
        return h

    def evaluate2(self):  # blue is enemy
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if self.board[row][col].is_blue():
                        if self.board[row][col].is_king():
                            h -= 10
                        else:
                            h -= 1
                    else:
                        if self.board[row][col].is_king():
                            h += 10
                        else:
                            h += 1
        return h

    def _integrity(self, is_blue_turn: bool) -> int:
        sizes = []
        elements = {}
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if (is_blue_turn and self.board[row][col].is_blue()) or (
                            not is_blue_turn and self.board[row][col].is_white()):
                        elements[(row, col)] = False

        while False in elements.values():
            for coords, visited in elements.items():
                if visited is False:
                    self._visit_node(elements, coords)
                    number_of_visited = len(list(filter(lambda element: element is True, elements.values())))
                    if len(sizes) == 0:
                        sizes.append(number_of_visited)
                    else:
                        sizes.append(number_of_visited - sum(sizes))
                    break
        return max(sizes)

    def _visit_node(self, elements: {(int, int): bool}, coords: (int, int)):
        elements[coords] = True
        for coord in self._get_neighbour_coords(coords):
            if coord in elements and elements[coord] is False:
                self._visit_node(elements, coord)

    @staticmethod
    def _get_neighbour_coords(coords: (int, int)):
        ret_coords = []

        if coords[0] > 0:
            if coords[1] > 0:
                ret_coords.append((coords[0]-1, coords[1]-1))
            if coords[1] < BOARD_WIDTH-1:
                ret_coords.append((coords[0] - 1, coords[1] + 1))
        if coords[0] < BOARD_WIDTH-1:
            if coords[1] > 0:
                ret_coords.append((coords[0]+1, coords[1]-1))
            if coords[1] < BOARD_WIDTH-1:
                ret_coords.append((coords[0] + 1, coords[1] + 1))

        return ret_coords

    def evaluate2_1(self):  # blue is enemy
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if self.board[row][col].is_blue():
                        if self.board[row][col].is_king():
                            h -= 10
                        else:
                            h -= 1
                    else:
                        if self.board[row][col].is_king():
                            h += 10
                        else:
                            h += 1
        return h + self._integrity(False)

    def evaluate2_2(self):  # blue is enemy
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if self.board[row][col].is_blue():
                        if self.board[row][col].is_king():
                            h -= 10
                        else:
                            h -= 1
                    else:
                        if self.board[row][col].is_king():
                            h += 10
                        else:
                            if row >= BOARD_WIDTH / 2:
                                h += 5
                            else:
                                h += 7
        return h

    def evaluate2_3(self):  # blue is enemy
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if self.board[row][col].is_blue():
                        if self.board[row][col].is_king():
                            h -= 10
                        else:
                            h -= 1
                    else:
                        if self.board[row][col].is_king():
                            h += 10
                        else:
                            h += 5 + (BOARD_WIDTH - (row + 1))
        return h

    def get_possible_moves(self, is_blue_turn):
        pos_moves = []
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if (is_blue_turn and self.board[row][col].is_blue()) or (
                            not is_blue_turn and self.board[row][col].is_white()):
                        pos_moves.extend(self.get_piece_moves(self.board[row][col]))
        return pos_moves

    def draw(self):
        self.window.fill(WHITE)
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                y = row * FIELD_SIZE
                x = col * FIELD_SIZE
                pygame.draw.rect(self.window, BLACK, (x, y, FIELD_SIZE, FIELD_SIZE))
                self.board[row][col].draw()

    def move(self, field):
        d_row = field.row
        d_col = field.col
        row_from = field.row_from
        col_from = field.col_from
        self.board[row_from][col_from].toogle_mark()
        self.something_is_marked = False
        self.board[d_row][d_col] = self.board[row_from][col_from]
        self.board[d_row][d_col].row = d_row
        self.board[d_row][d_col].col = d_col
        self.board[row_from][col_from] = Field()

        if field.pos_move.captures:
            fig_to_del = field.pos_move.captures

            self.board[fig_to_del.row][fig_to_del.col] = Field()
            if self.white_turn:
                self.blue_fig_left -= 1
            else:
                self.white_fig_left -= 1

        if self.white_turn and d_row == 0:  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        if not self.white_turn and d_row == BOARD_WIDTH - 1:  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        self.white_turn = not self.white_turn

    def end(self):
        return self.white_fig_left == 0 or self.blue_fig_left == 0 or len(
            self.get_possible_moves(not self.white_turn)) == 0

    def clicked_at(self, row, col):
        field = self.board[row][col]
        if field.is_move_mark():
            self.move(field)
        if (field.is_white() and self.white_turn and not self.something_is_marked) or (
                field.is_blue() and not self.white_turn and not self.something_is_marked):
            field.toogle_mark()
            self.something_is_marked = True
        elif self.something_is_marked and field.is_marked():
            field.toogle_mark()
            self.something_is_marked = False

    # tu spore powtorzenie kodu z move
    def make_ai_move(self, move):
        d_row = move.dest_row
        d_col = move.dest_col
        row_from = move.piece.row
        col_from = move.piece.col

        self.board[d_row][d_col] = self.board[row_from][col_from]
        self.board[d_row][d_col].row = d_row
        self.board[d_row][d_col].col = d_col
        self.board[row_from][col_from] = Field()

        if move.captures:
            fig_to_del = move.captures

            self.board[fig_to_del.row][fig_to_del.col] = Field()
            if self.white_turn:
                self.blue_fig_left -= 1
            else:
                self.white_fig_left -= 1

        if self.white_turn and d_row == 0:  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        if not self.white_turn and d_row == BOARD_WIDTH - 1:  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        self.white_turn = not self.white_turn


class Game:
    def __init__(self, window):
        self.window = window
        self.board = Board(window)

    def update(self):
        self.board.draw()
        pygame.display.update()

    def mouse_to_indexes(self, pos):
        return int(pos[0] // FIELD_SIZE), int(pos[1] // FIELD_SIZE)

    def clicked_at(self, pos):
        (col, row) = self.mouse_to_indexes(pos)
        self.board.clicked_at(row, col)


def prep_concurrent(move: Move, board: Board, depth: int) -> (Move, int or float):
    _board = deepcopy(board)
    _board.make_ai_move(move)
    return move, minimax_a_b_recurr(_board, depth - 1, False)


def prep_concurrent2(move: Move, board: Board, depth: int) -> (Move, int or float):
    _board = deepcopy(board)
    _board.make_ai_move(move)
    return move, minimax_a_b_recurr2(_board, depth - 1, False)


# ToDo
def minimax_a_b(board: Board, depth: int) -> Move:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = executor.map(lambda _move: prep_concurrent(_move, board, depth), board.get_possible_moves(True))

    hmm = tuple(result)
    best_move = max(hmm, key=lambda _move: _move[1])

    return best_move[0]


def minimax_a_b2(board: Board, depth: int) -> Move:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = executor.map(lambda _move: prep_concurrent2(_move, board, depth), board.get_possible_moves(False))

    hmm = tuple(result)
    best_move = max(hmm, key=lambda _move: _move[1])

    return best_move[0]


# ToDo
def minimax_a_b_recurr(board: Board, depth: int, move_max: bool, a=-math.inf, b=math.inf) -> int or float:
    if 0 == depth:
        return board.evaluate()
    if board.white_fig_left == 0:
        return math.inf
    if board.blue_fig_left == 0:
        return -math.inf

    moves = board.get_possible_moves(move_max)

    if move_max:
        for move in moves:
            _board = deepcopy(board)
            _board.make_ai_move(move)
            a = max(a, minimax_a_b_recurr(_board, depth - 1, not move_max, a, b))
            if a >= b:
                return b
        return a
    else:
        for move in moves:
            _board = deepcopy(board)
            _board.make_ai_move(move)
            b = min(b, minimax_a_b_recurr(_board, depth - 1, not move_max, a, b))
            if a >= b:
                return a
        return b


def minimax_a_b_recurr2(board: Board, depth: int, move_max: bool, a=-math.inf, b=math.inf) -> int or float:
    if 0 == depth:
        return board.evaluate2_1()
    if board.white_fig_left == 0:
        return -math.inf
    if board.blue_fig_left == 0:
        return math.inf

    moves = board.get_possible_moves(not move_max)

    if move_max:
        for move in moves:
            _board = deepcopy(board)
            _board.make_ai_move(move)
            a = max(a, minimax_a_b_recurr2(_board, depth - 1, not move_max, a, b))
            if a >= b:
                return b
        return a
    else:
        for move in moves:
            _board = deepcopy(board)
            _board.make_ai_move(move)
            b = min(b, minimax_a_b_recurr2(_board, depth - 1, not move_max, a, b))
            if a >= b:
                return a
        return b


def main():
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    is_running = True
    clock = pygame.time.Clock()
    game = Game(window)

    while is_running:
        clock.tick(FPS)

        if game.board.end():

            is_running = False

            break  # przydalby sie jakiś komunikat kto wygrał zamiast break

        if not game.board.white_turn:
            move = minimax_a_b(deepcopy(game.board), MINIMAX_DEPTH)
            game.board.make_ai_move(move)
        # else:
        #     move = minimax_a_b2(deepcopy(game.board), MINIMAX_DEPTH)
        #     game.board.make_ai_move(move)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                game.clicked_at(pos)

        game.update()

    pygame.quit()


if __name__ == '__main__':
    main()
