import chess
import chess
import chess.engine

from chess.engine import Cp, Mate


async def play_move(board, path_prefix):
    transport, engine = await chess.engine.popen_uci(path_prefix + "stockfish_13_win_x64_bmi2/stockfish_13_win_x64_bmi2.exe")

    result = await engine.play(board, chess.engine.Limit(depth=18))
    await engine.quit()
    return result.move


async def evaluate_position(board, path_prefix):

    # need to add this to root directory (not local)
    transport, engine = await chess.engine.popen_uci(path_prefix + "stockfish_13_win_x64_bmi2/stockfish_13_win_x64_bmi2.exe")

    info = await engine.analyse(board, chess.engine.Limit(depth=18))
    # x = info["score"].white()
    # y = info["score"].black()
    score = str(info["score"])
    # model = 'lichess'
    x = 2 * info["score"].white().wdl().expectation() - 1
    score_index = 0
    try:
        score_index = score.index("+")
    except ValueError:
        pass
    else:
        score_index = score_index

    try:
        score_index = score.index("-")
    except ValueError:
        pass
    else:
        score_index = score_index

    if score_index != 0:
        s = score[score_index]
    else:
        s = ""
    for i in range(score_index + 1, len(score)):
        curr_char = score[i]
        try:
            int(curr_char)
        except ValueError:
            continue
        else:
            s += curr_char

    s = int(s)

    if "Mate" in score:
        s = 10000 - s if s > 0 else -10000 - s

    await engine.quit()
    # return probabilty, actual score (-1 - 0) (-10000, +10000)
    return x, -s
