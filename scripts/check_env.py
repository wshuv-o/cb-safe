from __future__ import annotations

import csv
import sys
import time
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crypto import KEM, KEMS, details, is_enabled, liboqs_version  # noqa: E402

ALGS = list(KEMS)  # all logical KEM names
N = 30  # repetitions for median timing


def bench(name: str, n: int = N) -> dict:
    kg, en, de = [], [], []
    ok = True
    for _ in range(n):
        with KEM(name) as party:
            t = time.perf_counter(); pk = party.keygen(); kg.append((time.perf_counter() - t) * 1e3)
            t = time.perf_counter(); ct, ss_enc = KEM.encapsulate(name, pk); en.append((time.perf_counter() - t) * 1e3)
            t = time.perf_counter(); ss_dec = party.decapsulate(ct); de.append((time.perf_counter() - t) * 1e3)
            ok = ok and (ss_enc == ss_dec)
    d = details(name)
    d.update(
        roundtrip_ok=ok,
        keygen_ms=round(st.median(kg), 4),
        encap_ms=round(st.median(en), 4),
        decap_ms=round(st.median(de), 4),
    )
    return d


def main() -> int:
    print(f"liboqs {liboqs_version()}")
    enabled = {a: is_enabled(a) for a in ALGS}
    missing = [a for a, ok in enabled.items() if not ok]
    if missing:
        print(f"WARNING: not compiled into this liboqs build: {missing}")

    rows = [bench(a) for a in ALGS if enabled[a]]

    cols = ["name", "family", "claimed_nist_level", "public_key_bytes", "ciphertext_bytes",
            "shared_secret_bytes", "roundtrip_ok", "keygen_ms", "encap_ms", "decap_ms"]
    hdr = f"{'algorithm':12s} {'fam':10s} {'L':>1s} {'pk':>6s} {'ct':>6s} {'ss':>3s} {'ok':>3s} {'kg(ms)':>8s} {'en(ms)':>8s} {'de(ms)':>8s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:12s} {r['family']:10s} {r['claimed_nist_level']:>1d} "
              f"{r['public_key_bytes']:>6d} {r['ciphertext_bytes']:>6d} {r['shared_secret_bytes']:>3d} "
              f"{'OK' if r['roundtrip_ok'] else 'FAIL':>3s} "
              f"{r['keygen_ms']:>8.3f} {r['encap_ms']:>8.3f} {r['decap_ms']:>8.3f}")

    out = ROOT / "results" / "crypto_kem_bench.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"\nwrote {out}")

    all_ok = all(r["roundtrip_ok"] for r in rows) and not missing
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
