import chess
import random
import tensorflow as tf


class Minimax:

    def __init__(self, depth, board, is_max, value_network=None):
        self.depth, self.board, self.is_max = depth, board, is_max
        self.eval_network = value_network
        # print("board", self.board)

    def __run(self):
        # init, get all legal moves from chess board
        legal_actions = self.board.legal_moves
        # curr action score is -inf (so that all found action scores are higher than init min value)
        action_score = float("-Inf")
        best_final_action = None
        # iterate over legal actions
        for san in legal_actions:
            # get move in san form
            action = chess.Move.from_uci(str(san))
            # push move onto board stack
            self.board.push(action)
            # recursively call minimax with d = d - 1, curr board, init alpha / beta values and if to maximize
            value = max(action_score, self.__minimax(self.depth - 1, self.board, float("-Inf"), float("Inf"), not self.is_max))
            # remove move from board stack (we have found its value at depth d)
            self.board.pop()
            # if the action_score is better than curr best action score, update
            if value > action_score:
                # we can print new best score / value (testing)
                action_score = value
                best_final_action = action
        return best_final_action

    def __minimax(self, depth, board, alpha, beta, is_maximizing):
        # if depth = 0, we reached max depth (d) return board evaluation
        if depth == 0:
            return -self.__evaluation(board)
        possibleMoves = board.legal_moves
        if is_maximizing:
            bestMove = float("-Inf")
            for san in possibleMoves:
                move = chess.Move.from_uci(str(san))
                board.push(move)
                bestMove = max(bestMove, self.__minimax(depth - 1, board, alpha, beta, not is_maximizing))
                board.pop()
                alpha = max(alpha, bestMove)
                if beta <= alpha:
                    return bestMove
            return bestMove
        else:
            bestMove = float("Inf")
            for san in possibleMoves:
                move = chess.Move.from_uci(str(san))
                board.push(move)
                bestMove = min(bestMove, self.__minimax(depth - 1, board, alpha, beta, not is_maximizing))
                board.pop()
                beta = min(beta, bestMove)
                if beta <= alpha:
                    return bestMove
            return bestMove

    def get_move(self):
        return self.__run()

    def __evaluation(self, board):
        if self.eval_network:

            # might need to cast from tf.tensor -> int
            v = self.eval_network(board, minimax=True)
            return v
        # random eval for now
        return random.random()




