from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OQS_INSTALL_PATH", str(Path.home() / "_oqs"))

def _oqs():
    import oqs
    return oqs

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

FAMILY = {name: ("code-based" if name.startswith("hqc") else "lattice") for name in KEMS}


class UnknownKEM(KeyError):
    pass


def mechanism(name: str) -> str:
    try:
        return KEMS[name]
    except KeyError:
        raise UnknownKEM(f"{name!r}; known: {sorted(KEMS)}") from None


def is_enabled(name: str) -> bool:
    return mechanism(name) in _oqs().get_enabled_kem_mechanisms()


def details(name: str) -> dict:
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

    def __init__(self, name: str):
        self.name = name
        self.mechanism = mechanism(name)
        self._kem = _oqs().KeyEncapsulation(self.mechanism)
        self._public_key: bytes | None = None

    def keygen(self) -> bytes:
        self._public_key = self._kem.generate_keypair()
        return self._public_key

    @property
    def public_key(self) -> bytes:
        if self._public_key is None:
            raise RuntimeError("call keygen() first")
        return self._public_key

    def decapsulate(self, ciphertext: bytes) -> bytes:
        return self._kem.decap_secret(ciphertext)

    @staticmethod
    def encapsulate(name: str, public_key: bytes) -> tuple[bytes, bytes]:
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
