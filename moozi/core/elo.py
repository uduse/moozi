INIT = 1300


def expected(a: float, b: float):
    return 1 / (1 + 10 ** ((b - a) / 400))


def elo(old: float, exp: float, score: float, k: int = 32):
    return old + k * (score - exp)


def update(a: float, b: float, result: float, k=32):
    assert 0 <= result <= 1
    updated_a = elo(a, expected(a, b), result, k=k)
    updated_b = elo(b, expected(b, a), 1 - result, k=k)
    return (updated_a, updated_b)
