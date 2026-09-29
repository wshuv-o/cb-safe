"""Scalability of per-client cost with the client population N.

Secure-aggregation cost scales with the number of pairwise partners each client
maintains. All-pairs secure aggregation (Bonawitz) gives every client N-1
partners -> O(N). CB-SAFE masks only *within clusters of size c*, capping
partners at c-1, so per-client setup traffic and per-round mask compute are
CONSTANT in N. Curves use the measured HQC-128 / ML-KEM-512 primitives at c=3
(results/overhead.csv), extrapolated by partner count. Writes fig_scaling_cost.
"""

import _bootstrap  # noqa: F401

import os
import numpy as np
import pandas as pd

import plot_style as st
st.apply()
import matplotlib.pyplot as plt  # noqa: E402

R = _bootstrap.RESULTS
FIGS = os.path.join(R, "figs")
C = 3                                   # cluster size / anonymity set
NS = np.array([30, 100, 300, 1000, 3000])

o = pd.read_csv(os.path.join(R, "overhead.csv"))
o = o[o.cluster_size == C].set_index("kem")


def per_partner(kem, col):
    # measured cost at c=3 covers (c-1) partners; divide out to get per-partner
    return o.loc[kem, col] / (C - 1)


# --- per-client one-time setup traffic (KiB) ---
hqc_setup_pp = (per_partner("hqc-128", "setup_up_B") + per_partner("hqc-128", "setup_down_B")) / 1024
mlk_setup_pp = (per_partner("mlkem-512", "setup_up_B") + per_partner("mlkem-512", "setup_down_B")) / 1024
# --- per-client per-round mask-derivation compute (ms) ---
hqc_mask_pp = per_partner("hqc-128", "round_client_mask_s") * 1000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(st.COL_DOUBLE, 2.7))

# Panel (a): setup traffic
ax1.loglog(NS, hqc_setup_pp * (NS - 1), color=st.style("mean")["color"], lw=1.6,
           marker="^", ms=4, label="All-pairs SecAgg (HQC), $O(N)$")
ax1.loglog(NS, np.full_like(NS, hqc_setup_pp * (C - 1), dtype=float), color="#0b0b0b",
           lw=2.2, marker="o", ms=4, label="CB-SAFE cluster (HQC, $c{=}3$)")
ax1.loglog(NS, np.full_like(NS, mlk_setup_pp * (C - 1), dtype=float),
           color=st.style("fedgt")["color"], lw=1.6, ls=(0, (4, 2)), marker="s", ms=4,
           label="CB-SAFE cluster (ML-KEM, $c{=}3$)")
ax1.set_xlabel("clients $N$")
ax1.set_ylabel("per-client setup traffic (KiB)")
sav = hqc_setup_pp * (1000 - 1) / (hqc_setup_pp * (C - 1))
ax1.annotate(f"${sav:.0f}\\times$ less\nat $N{{=}}1000$", (1000, hqc_setup_pp * (C - 1)),
             textcoords="offset points", xytext=(-8, 10), fontsize=6.5, color="#6b6b6b", ha="right")

# Panel (b): per-round mask compute
ax2.loglog(NS, hqc_mask_pp * (NS - 1), color=st.style("mean")["color"], lw=1.6,
           marker="^", ms=4, label="All-pairs SecAgg, $O(N)$")
ax2.loglog(NS, np.full_like(NS, hqc_mask_pp * (C - 1), dtype=float), color="#0b0b0b",
           lw=2.2, marker="o", ms=4, label="CB-SAFE cluster ($c{=}3$)")
ax2.set_xlabel("clients $N$")
ax2.set_ylabel("per-client mask compute / round (ms)")

for ax in (ax1, ax2):
    ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
    ax.grid(True, which="both", axis="both", color="#e1e0d9", lw=0.5)
ax1.legend(fontsize=6, frameon=False, loc="upper left")
ax2.legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout()
out = os.path.join(FIGS, "fig_scaling_cost")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".pdf/.png")

# --- companion table (markdown preview + LaTeX) ---
rows = []
for N in NS:
    naive = hqc_setup_pp * (N - 1)
    rows.append((N, naive / 1024, hqc_setup_pp * (C - 1), mlk_setup_pp * (C - 1), naive / (hqc_setup_pp * (C - 1))))
print("\nN     naive-HQC(MiB)  CBSAFE-HQC(KiB)  CBSAFE-MLKEM(KiB)  savings x")
for N, nm, hk, mk, sv in rows:
    print(f"{N:5d}  {nm:12.2f}  {hk:14.1f}  {mk:16.1f}  {sv:8.0f}")
