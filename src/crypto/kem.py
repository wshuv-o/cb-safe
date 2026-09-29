"""Thin, KEM-agnostic wrapper over liboqs (Open Quantum Safe).

The rest of the framework talks to KEMs only through this module, so the secure
aggregation protocol never hard-codes whether the post-quantum primitive is the
code-based HQC or the lattice-based Kyber / ML-KEM. Swapping the KEM is a one-line
config change (the whole point of the "cryptographic diversity" contribution).

Logical names (lower-case, used in configs) map to liboqs mechanism strings:

    hqc-128 / hqc-192 / hqc-256          -> HQC-*        (code-based, the contribution)
    mlkem-512 / mlkem-768 / mlkem-1024   -> ML-KEM-*     (the standardized Kyber)
    kyber-512 / kyber-768 / kyber-1024   -> Kyber*       (round-3 Kyber, lattice baseline)

ML-KEM is the FIPS-203 standardization of Kyber; prefer mlkem-* for the lattice baseline
and keep kyber-* available for direct comparison with older literature.
"""

from __future__ import annotations

import os
from pathlib import Path

# Ensure a locally built liboqs (see README) is discoverable before importing oqs.
# On Linux (Colab/Kaggle) liboqs-python auto-builds and this is simply ignored.
os.environ.setdefault("OQS_INSTALL_PATH", str(Path.home() / "_oqs"))

# liboqs is imported lazily so that importing this module (and hence the whole
# aggregation/simulation stack) does NOT require a liboqs build. The robustness
# experiments never instantiate a KEM; only the overhead/equivalence experiments
# do, and they run where liboqs is installed. This keeps the pure-PyTorch Kaggle
# runner dependency-free.
def _oqs():
    import oqs  # noqa: PLC0415
    return oqs

# Logical name -> liboqs mechanism string.
KEMS: dict[str, str] = {
    "hqc-128": "HQC-128",
    "hqc-192": "HQC-192",
    "hqc-256": "HQC-256",
    "mlkem-512": "ML-KEM-512",
    "mlkem-768": "ML-KEM-768",
    "mlkem-1024": "ML-KEM-1024",
    "kyber-512": "Kyber512",
    "kyber-768": "Kyber768",
    "kyber-1024": "Kyber1024",
}

# Which family a logical name belongs to (for grouping / reporting).
FAMILY = {name: ("code-based" if name.startswith("hqc") else "lattice") for name in KEMS}


class UnknownKEM(KeyError):
    """Raised for a logical KEM name not in KEMS."""


def mechanism(name: str) -> str:
    """Resolve a logical name (e.g. 'hqc-128') to a liboqs mechanism (e.g. 'HQC-128')."""
    try:
        return KEMS[name]
    except KeyError:
        raise UnknownKEM(f"{name!r}; known: {sorted(KEMS)}") from None


def is_enabled(name: str) -> bool:
    """True if this KEM is compiled into the loaded liboqs build."""
    return mechanism(name) in _oqs().get_enabled_kem_mechanisms()


def details(name: str) -> dict:
    """Sizes + NIST level for a KEM, without generating any keys."""
    with _oqs().KeyEncapsulation(mechanism(name)) as kem:
        d = kem.details
    return {
        "name": name,
        "mechanism": d["name"],
        "family": FAMILY[name],
        "claimed_nist_level": d["claimed_nist_level"],
        "public_key_bytes": d["length_public_key"],
        "secret_key_bytes": d["length_secret_key"],
        "ciphertext_bytes": d["length_ciphertext"],
        "shared_secret_bytes": d["length_shared_secret"],
    }


class KEM:
    """One party's handle on a KEM keypair.

    Usage in secure aggregation:
        with KEM("hqc-128") as me:
            pk = me.keygen()                  # publish pk
            ct, ss = KEM.encapsulate("hqc-128", peer_pk)   # send ct to peer
            ss == me.decapsulate(peer_ct)     # recover the same shared secret
    """

    def __init__(self, name: str):
        self.name = name
        self.mechanism = mechanism(name)
        self._kem = _oqs().KeyEncapsulation(self.mechanism)
        self._public_key: bytes | None = None

    def keygen(self) -> bytes:
        """Generate a keypair; return the public key. Secret key stays inside this object."""
        self._public_key = self._kem.generate_keypair()
        return self._public_key

    @property
    def public_key(self) -> bytes:
        if self._public_key is None:
            raise RuntimeError("call keygen() first")
        return self._public_key

    def decapsulate(self, ciphertext: bytes) -> bytes:
        """Recover the shared secret from a ciphertext encapsulated to our public key."""
        return self._kem.decap_secret(ciphertext)

    @staticmethod
    def encapsulate(name: str, public_key: bytes) -> tuple[bytes, bytes]:
        """Encapsulate to someone else's public key. Returns (ciphertext, shared_secret)."""
        with _oqs().KeyEncapsulation(mechanism(name)) as enc:
            return enc.encap_secret(public_key)

    def free(self) -> None:
        self._kem.free()

    def __enter__(self) -> "KEM":
        return self

    def __exit__(self, *exc) -> None:
        self.free()


def liboqs_version() -> str:
    return _oqs().oqs_version()
