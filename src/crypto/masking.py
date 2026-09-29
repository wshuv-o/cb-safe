"""Fixed-point quantization and cryptographic mask generation for secure aggregation.

Masking arithmetic is exact: float32 update vectors are quantized to fixed-point
integers mod 2**64, masks are uniform uint64 keystream from ChaCha20, and pairwise
masks cancel exactly on summation. The only error versus plain float aggregation is
the quantization step (~2**-SCALE_BITS per coordinate), which run_utility.py verifies.

Per-round mask seeds are derived with HKDF-SHA256 from the long-lived KEM shared
secret and the round index, so key encapsulation happens once at setup and masks
refresh every round (the cross-round amortization the paper claims).
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

SCALE_BITS = 16
SCALE = 1 << SCALE_BITS


def quantize(x: np.ndarray) -> np.ndarray:
    """float32/64 vector -> fixed-point uint64 (two's complement mod 2**64)."""
    q = np.round(x.astype(np.float64) * SCALE).astype(np.int64)
    return q.astype(np.uint64)


def dequantize(s: np.ndarray) -> np.ndarray:
    """uint64 modular sum -> float64. Valid while |true scaled sum| < 2**63."""
    return s.astype(np.int64).astype(np.float64) / SCALE


def hkdf(key: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-SHA256 (extract with fixed salt + single-block expand chain)."""
    prk = hmac.new(b"cb-safe-hkdf-salt", key, hashlib.sha256).digest()
    okm, block = b"", b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


def pair_seed(shared_secret: bytes, round_idx: int) -> bytes:
    """Per-round pairwise mask seed from the long-lived KEM shared secret."""
    return hkdf(shared_secret, b"pairwise|round=%d" % round_idx)


def prg_mask(seed: bytes, length: int) -> np.ndarray:
    """Uniform uint64 vector of `length` from a ChaCha20 keystream keyed by seed."""
    key = hashlib.sha256(b"cb-safe-prg|" + seed).digest()
    cipher = Cipher(algorithms.ChaCha20(key, b"\x00" * 16), mode=None)
    keystream = cipher.encryptor().update(b"\x00" * (8 * length))
    return np.frombuffer(keystream, dtype=np.uint64).copy()
