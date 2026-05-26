def to_ping(r, c):
    """Biến (0,0) thành A1, (1, 12) thành B13"""
    return f"{chr(65 + c)}{r + 1}"