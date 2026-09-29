"""Shamir t-of-n secret sharing over GF(2**127 - 1) for 15-byte mask seeds.

Used for dropout recovery: each client shares its per-round self-mask seed b_i with
its cluster peers. The server reconstructs b_i for *surviving* clients (to remove
their self-masks) and never learns both a client's self-mask and its pairwise seeds,
which is the standard Bonawitz-style unmasking argument.

Secrets are 15 bytes (120 bits) so they always fit below the Mersenne prime 2**127-1.
"""

from __future__ import annotations

import secrets

P = (1 << 127) - 1  # Mersenne prime M127

SECRET_BYTES = 15
SHARE_BYTES = 1 + 16  # x coordinate (1 byte) + y coordinate (16 bytes)


def _eval_poly(coeffs: list[int], x: int) -> int:
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % P
    return acc


def new_secret() -> bytes:
    return secrets.token_bytes(SECRET_BYTES)


def split(secret: bytes, n: int, t: int) -> list[tuple[int, int]]:
    """Split a <=15-byte secret into n shares, any t of which reconstruct it."""
    if len(secret) > SECRET_BYTES:
        raise ValueError(f"secret must be <= {SECRET_BYTES} bytes")
    s = int.from_bytes(secret, "big")
    coeffs = [s] + [secrets.randbelow(P) for _ in range(t - 1)]
    return [(x, _eval_poly(coeffs, x)) for x in range(1, n + 1)]


def reconstruct(shares: list[tuple[int, int]]) -> bytes:
    """Lagrange interpolation at 0 over GF(P)."""
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
