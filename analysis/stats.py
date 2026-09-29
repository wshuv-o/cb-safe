import _bootstrap  # noqa: F401

import glob
import os
import re

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

R = _bootstrap.RESULTS
ROOTS = {
    "cifar10": R,
    "fmnist": os.path.join(R, "fmnist"),
    "emnist": os.path.join(R, "kaggle", "emnist"),
    "edgeiiot": os.path.join(R, "kaggle", "edgeiiot"),
}
PAT = re.compile(r"robust_(\w+?)_(\w+?)_f(\d+)_c(\d+)_s(\d+)\.csv$")


def load_long() -> pd.DataFrame:
    rows = []
    for dataset, root in ROOTS.items():
        if not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "robust_*.csv")):
            m = PAT.search(os.path.basename(path))
            if not m:
                continue
            attack, agg, f, c, seed = m[1], m[2], int(m[3]) / 100, int(m[4]), int(m[5])
            df = pd.read_csv(path)
            tail = df.tail(5)
            row = {"dataset": dataset, "attack": attack, "agg": agg, "f": f, "c": c,
                   "seed": seed, "final_acc": tail["acc"].mean()}
            if "asr" in df.columns:
                row["final_asr"] = tail["asr"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["dataset", "attack", "agg", "f", "c"])
    out = g.agg(
        acc_mean=("final_acc", "mean"),
        acc_std=("final_acc", "std"),
        n=("final_acc", "size"),
    ).reset_index()
    if "final_asr" in long.columns:
        asr = g.agg(asr_mean=("final_asr", "mean"), asr_std=("final_asr", "std")).reset_index()
        out = out.merge(asr, on=["dataset", "attack", "agg", "f", "c"], how="left")
    out = out.round(4)
    out.to_csv(os.path.join(R, "summary_multiseed.csv"), index=False)
    return out


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def significance(long: pd.DataFrame, attack: str = "signflip") -> pd.DataFrame | None:
    """CB-SAFE+ (reputation) vs each static rule. Pairs by (dataset, f, seed) so each
    comparison is like-for-like. Run for both sign-flip and label-flip."""
    # FLTrust is included so that the strongest non-collapsing baseline is tested,
    # not only the coordinate-wise rules that collapse under laundering.
    baselines = ["mean", "trimmed", "median", "krum", "bulyan", "geomedian", "fltrust"]
    sub = long[(long["attack"] == attack) & (long["c"] == 3)
               & long["dataset"].isin(["cifar10", "fmnist", "emnist", "edgeiiot"])]
    rep = sub[sub["agg"] == "reputation"].set_index(["dataset", "f", "seed"])["final_acc"]
    if rep.empty:
        return None
    results, pvals = [], []
    for base in baselines:
        b = sub[sub["agg"] == base].set_index(["dataset", "f", "seed"])["final_acc"]
        paired = pd.concat([rep.rename("rep"), b.rename("base")], axis=1).dropna()
        if len(paired) < 2:
            continue
        diff = paired["rep"] - paired["base"]
        rec = {"baseline": base, "n_pairs": len(paired),
               "cbsafe_mean": round(paired["rep"].mean(), 4),
               "base_mean": round(paired["base"].mean(), 4),
               "mean_gain": round(diff.mean(), 4)}
        if HAVE_SCIPY and diff.std() > 0:
            rec["t_p"] = float(sps.ttest_rel(paired["rep"], paired["base"]).pvalue)
            try:
                rec["wilcoxon_p"] = float(sps.wilcoxon(diff).pvalue)
            except ValueError:
                rec["wilcoxon_p"] = float("nan")
        else:
            rec["t_p"] = float("nan")
            rec["wilcoxon_p"] = float("nan")
        results.append(rec)
        pvals.append(rec["t_p"] if not np.isnan(rec.get("t_p", np.nan)) else 1.0)
    if not results:
        return None
    for rec, padj in zip(results, holm(pvals)):
        rec["t_p_holm"] = round(padj, 5)
        rec["t_p"] = round(rec["t_p"], 5)
        rec["wilcoxon_p"] = round(rec["wilcoxon_p"], 5)
    out = pd.DataFrame(results)
    name = "significance.csv" if attack == "signflip" else f"significance_{attack}.csv"
    out.to_csv(os.path.join(R, name), index=False)
    return out


def main() -> None:
    long = load_long()
    if long.empty:
        print("no results found yet")
        return
    print(f"loaded {len(long)} runs across datasets: "
          f"{sorted(long['dataset'].unique())}, seeds {sorted(long['seed'].unique())}")
    summ = summarize(long)
    # per-dataset headline: CB-SAFE+ vs the best static rule at each (dataset, attack, f)
    print("\n== per-dataset: CB-SAFE+ vs best static rule (sign-flip) ==")
    sf = long[(long["attack"] == "signflip") & (long["c"] == 3)]
    for ds in sorted(sf["dataset"].unique()):
        d = sf[sf["dataset"] == ds]
        for f in sorted(d["f"].unique()):
            cell = d[d["f"] == f]
            rep = cell[cell["agg"] == "reputation"]["final_acc"].mean()
            statics = cell[cell["agg"] != "reputation"]
            if statics.empty or cell[cell["agg"] == "reputation"].empty:
                continue
            best = statics.groupby("agg")["final_acc"].mean()
            print(f"  {ds:9s} f={f:.1f}: CB-SAFE+={rep:.3f}  best-static={best.max():.3f} "
                  f"({best.idxmax()})  worst-static={best.min():.3f}")
    written = ["summary_multiseed.csv"]
    for attack in ("signflip", "labelflip"):
        sig = significance(long, attack)
        if sig is None:
            continue
        print(f"\n== CB-SAFE+ vs baselines ({attack}, paired, Holm-corrected) ==")
        print(sig.to_string(index=False))
        written.append("significance.csv" if attack == "signflip"
                       else f"significance_{attack}.csv")
    print("\nwrote " + ", ".join(written))


if __name__ == "__main__":
    main()
