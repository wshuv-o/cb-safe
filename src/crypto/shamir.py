from __future__ import annotations

import secrets

P = (1 << 127) - 1

SECRET_BYTES = 15
SHARE_BYTES = 1 + 16


def _eval_poly(coeffs: list[int], x: int) -> int:
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % P
    return acc


def new_secret() -> bytes:
    return secrets.token_bytes(SECRET_BYTES)


def split(secret: bytes, n: int, t: int) -> list[tuple[int, int]]:
    if len(secret) > SECRET_BYTES:
        raise ValueError(f"secret must be <= {SECRET_BYTES} bytes")
    s = int.from_bytes(secret, "big")
    coeffs = [s] + [secrets.randbelow(P) for _ in range(t - 1)]
    return [(x, _eval_poly(coeffs, x)) for x in range(1, n + 1)]


def reconstruct(shares: list[tuple[int, int]]) -> bytes:
    total = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i == j:
                continue
            num = (num * (-xj)) % P
            den = (den * (xi - xj)) % P
        total = (total + yi * num * pow(den, P - 2, P)) % P
    return total.to_bytes(SECRET_BYTES, "big")
