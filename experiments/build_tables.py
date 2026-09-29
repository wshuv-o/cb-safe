"""Build wide, multi-metric manuscript tables from all result CSVs.
Outputs markdown (preview) + LaTeX booktabs into results/tables/."""

import _bootstrap  # noqa: F401

import glob
import os
import re
import collections

import numpy as np
import pandas as pd

R = _bootstrap.RESULTS
TBL = os.path.join(R, "tables")
os.makedirs(TBL, exist_ok=True)
ROOTS = {"CIFAR-10": "", "FashionMNIST": "fmnist",
         "EMNIST": "kaggle/emnist", "Edge-IIoTset": "kaggle/edgeiiot"}
RULES = [("mean", "FedAvg mean"), ("trimmed", "Trimmed mean"), ("median", "Median"),
         ("krum", "Multi-Krum"), ("fedgt", "FedGT [TIFS'25]"),
         ("reputation", "CB-SAFE+"), ("hybrid_ov4", "CB-SAFE+ hybrid")]
FS = [0.1, 0.2, 0.3]


def acc(path):
    d = pd.read_csv(path).tail(5)
    return d["acc"].mean() * 100, (d["asr"].mean() * 100 if "asr" in d.columns else None)


def collect(dsdir, agg, attack):
    root = os.path.join(R, dsdir) if dsdir else R
    out = collections.defaultdict(list)
    # exact match so 'reputation' doesn't catch 'reputation_tf'/'_ov4'
    for p in glob.glob(os.path.join(root, f"robust_{attack}_{agg}_f*_c3_s*.csv")):
        b = os.path.basename(p)
        core = re.sub(r"_f\d+_c3_s\d+\.csv$", "", b).replace(f"robust_{attack}_", "")
        if core != agg:
            continue
        m = re.search(r"_f(\d+)_c3_s(\d+)", b)
        out[int(m[1]) / 100].append(p)
    return out


def cell(paths, asr=False):
    if not paths:
        return "--"
    vals = [acc(p)[1 if asr else 0] for p in paths]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    m, s = np.mean(vals), np.std(vals)
    return f"{m:.1f}\\pm{s:.1f}" if len(vals) > 1 else f"{m:.1f}"


# ---------- MASTER ROBUSTNESS TABLE ----------
rows = []
for ds, dsdir in ROOTS.items():
    for agg, label in RULES:
        sf = collect(dsdir, agg, "signflip")
        lf = collect(dsdir, agg, "labelflip")
        bd = collect(dsdir, agg, "backdoor")
        if not (sf or lf or bd):
            continue
        rows.append([ds, label] +
                    [cell(sf.get(f)) for f in FS] +
                    [cell(lf.get(f)) for f in FS] +
                    [cell(bd.get(f), asr=True) for f in FS])
cols = (["Dataset", "Aggregation"] +
        [f"SF f{int(f*100)}" for f in FS] +
        [f"LF f{int(f*100)}" for f in FS] +
        [f"BD-ASR f{int(f*100)}" for f in FS])
master = pd.DataFrame(rows, columns=cols)
master.to_csv(os.path.join(TBL, "master_robustness.csv"), index=False)
print("=== MASTER ROBUSTNESS (sign-flip / label-flip acc %, backdoor ASR %) ===")
print(master.to_string(index=False))

# ---------- OVERHEAD WIDE ----------
o = pd.read_csv(os.path.join(R, "overhead.csv"))
o = o[o.cluster_size == 3].copy()
o["break_even_rounds"] = ((o.setup_up_B + o.setup_down_B) / (o.round_up_B + o.round_down_B)).round(4)
ow = o[["kem", "setup_up_B", "setup_down_B", "setup_wall_s", "round_up_B",
        "round_client_mask_s", "round_server_unmask_s", "break_even_rounds"]]
ow.to_csv(os.path.join(TBL, "overhead_wide.csv"), index=False)
print("\n=== OVERHEAD WIDE (per client, c=3) ===")
print(ow.to_string(index=False))

# ---------- IDENTIFICATION QUALITY ----------
def ident(dsdir, agg):
    root = os.path.join(R, dsdir) if dsdir else R
    out = {}
    for f in FS:
        ps = glob.glob(os.path.join(root, f"robust_signflip_{agg}_f{int(f*100):02d}_c3_s*.csv"))
        ps = [p for p in ps if re.sub(r"_f\d+_c3_s\d+\.csv$", "", os.path.basename(p)).replace("robust_signflip_", "") == agg]
        tp, fp, nm = [], [], []
        for p in ps:
            d = pd.read_csv(p).iloc[-1]
            if "excluded_malicious" in d:
                tp.append(d["excluded_malicious"]); fp.append(d["excluded_honest"]); nm.append(d["n_malicious"])
        if tp:
            out[f] = f"{np.mean(tp):.0f}/{np.mean(nm):.0f}, FP {np.mean(fp):.1f}"
    return out

print("\n=== IDENTIFICATION QUALITY (CIFAR-10, attackers caught / total, false positives) ===")
for agg, label in [("fedgt", "FedGT"), ("reputation", "CB-SAFE+"), ("hybrid_ov4", "hybrid")]:
    iq = ident("", agg)
    if iq:
        print(f"  {label:10s} " + " | ".join(f"f{int(f*100)}: {iq[f]}" for f in FS if f in iq))

print("\nwrote results/tables/{master_robustness,overhead_wide}.csv")
