from __future__ import annotations

import hashlib
import hmac

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

SCALE_BITS = 16
SCALE = 1 << SCALE_BITS


def quantize(x: np.ndarray) -> np.ndarray:
    q = np.round(x.astype(np.float64) * SCALE).astype(np.int64)
    return q.astype(np.uint64)


def dequantize(s: np.ndarray) -> np.ndarray:
    return s.astype(np.int64).astype(np.float64) / SCALE


def hkdf(key: bytes, info: bytes, length: int = 32) -> bytes:
    prk = hmac.new(b"cb-safe-hkdf-salt", key, hashlib.sha256).digest()
    okm, block = b"", b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


def pair_seed(shared_secret: bytes, round_idx: int) -> bytes:
    return hkdf(shared_secret, b"pairwise|round=%d" % round_idx)


def prg_mask(seed: bytes, length: int) -> np.ndarray:
    key = hashlib.sha256(b"cb-safe-prg|" + seed).digest()
    cipher = Cipher(algorithms.ChaCha20(key, b"\x00" * 16), mode=None)
    keystream = cipher.encryptor().update(b"\x00" * (8 * length))
    return np.frombuffer(keystream, dtype=np.uint64).copy()
