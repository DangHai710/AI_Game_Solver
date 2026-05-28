def _valid(board, r, c, n):
    if any(board[r][i] == n or board[i][c] == n for i in range(9)):
        return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return all(board[i][j] != n for i in range(br, br + 3) for j in range(bc, bc + 3))


def is_board_valid(board):
    for values in _groups(board):
        nums = [int(v) for v in values if int(v) != 0]
        if len(nums) != len(set(nums)):
            return False
    return True


def _groups(board):
    for i in range(9):
        yield board[i, :]
        yield board[:, i]
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            yield board[r : r + 3, c : c + 3].ravel()


def _best_empty_cell(board):
    best = None
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                continue
            candidates = [n for n in range(1, 10) if _valid(board, r, c, n)]
            if not candidates:
                return r, c, []
            if best is None or len(candidates) < len(best[2]):
                best = (r, c, candidates)
    return best


def solve_sudoku(board):
    if not is_board_valid(board):
        return False

    cell = _best_empty_cell(board)
    if cell is None:
        return True

    r, c, candidates = cell
    for n in candidates:
        board[r][c] = n
        if solve_sudoku(board):
            return True
        board[r][c] = 0
    return False
