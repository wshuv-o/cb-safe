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
DSROOT = {"CIFAR-10": "", "FashionMNIST": "fmnist",
          "EMNIST": "kaggle/emnist", "Edge-IIoTset": "kaggle/edgeiiot"}
# dsdir -> dataset name used by the faithful-FedGT filenames
DSNAME = {"": "cifar10", "fmnist": "fmnist", "kaggle/emnist": "emnist",
          "kaggle/edgeiiot": "edgeiiot"}
FS = [0.1, 0.2, 0.3]
SEEDS_FINAL = 3          # a cell is "final" only with this many seeds at >=50 rounds


def runs(dsdir, agg, attack, f):
    # FedGT: read the faithful (BCJR) results, which live in results/fedgt_faithful/
    # under a different naming (faithful_fedgt_{ds}_{attack}_f{XX}_N30_s{seed}.csv).
    if agg == "fedgt":
        dsn = DSNAME.get(dsdir)
        pat = os.path.join(R, "fedgt_faithful",
                           f"faithful_fedgt_{dsn}_{attack}_f{int(f*100):02d}_N30_s*.csv")
        return [p for p in glob.glob(pat) if "oneshot" not in p]
    root = os.path.join(R, dsdir) if dsdir else R
    out = []
    for p in glob.glob(os.path.join(root, f"robust_{attack}_{agg}_f{int(f*100):02d}_c3_s*.csv")):
        core = re.sub(r"_f\d+_c3_s\d+\.csv$", "", os.path.basename(p)).replace(f"robust_{attack}_", "")
        if core == agg:
            out.append(p)
    return out


def val(paths, asr=False):
    vs = []
    rounds_ok = True
    for p in paths:
        full = pd.read_csv(p)
        if len(full) < 50:          # stale (e.g. old 30-round) or truncated run
            rounds_ok = False
        d = full.tail(5)
        col = "asr" if asr else "acc"
        if col in d.columns:
            vs.append(d[col].mean() * 100)
    if not vs:
        return None
    final = (len(vs) >= SEEDS_FINAL and rounds_ok)   # complete 3-seed, 50-round data
    return (np.mean(vs), np.std(vs), len(vs), final)


def _red(txt):
    return f"\\textcolor{{red}}{{{txt}}}"


LEGEND = (r" A dash ($-$) marks a configuration not yet run; such entries are "
          r"shown in red and are the only gaps in this table.")


def _legend_if_needed(cells):
    """Append the dash/red legend only when the table actually contains such a
    cell, so a completed grid never carries a legend for markup it lacks and a
    partial one is never left unexplained."""
    return LEGEND if any("textcolor{red}" in c for c in cells) else ""


def fmt(v, bold=False):
    if v is None:
        return _red("$-$")            # PROVISIONAL: experiment not run yet
    m, s, n = v[0], v[1], v[2]
    final = v[3] if len(v) > 3 else True
    txt = f"{m:.1f}\\,{{\\scriptsize$\\pm${s:.1f}}}" if n > 1 else f"{m:.1f}"
    if bold:
        txt = f"\\textbf{{{txt}}}"
    return txt if final else _red(txt)   # red = partial seeds / <50 rounds -> update later


def ident(dsdir, agg, f):
    root = os.path.join(R, dsdir) if dsdir else R
    tp, fp, nm = [], [], []
    for p in runs(dsdir, agg, "signflip", f):
        d = pd.read_csv(p).iloc[-1]
        if "excluded_malicious" in d:
            tp.append(d["excluded_malicious"]); fp.append(d["excluded_honest"]); nm.append(d["n_malicious"])
    if not tp:
        return "--"
    return f"{np.mean(tp):.0f}/{np.mean(nm):.0f}"


def identfp(dsdir, agg, f):
    root = os.path.join(R, dsdir) if dsdir else R
    fp = []
    for p in runs(dsdir, agg, "signflip", f):
        d = pd.read_csv(p).iloc[-1]
        if "excluded_honest" in d:
            fp.append(d["excluded_honest"])
    return f"{np.mean(fp):.1f}" if fp else "--"


# ============ sign-flip tables (full-width table*), reusable builder ============
def build_signflip_table(methods, title, caption, label, outfile, attack="signflip",
                         colsep="1pt"):
    # IEEE style: the caption is a short title-case title; the explanatory text is
    # set as a note under the table (\tablenote, defined in the manuscript preamble).
    # The red/dash legend is appended only when such cells actually exist. A fixed
    # legend went stale once the grid completed, explaining markup the table no
    # longer contained; hard-removing it would instead leave a future partial run
    # marked red with nothing explaining it.
    body, prev = _signflip_body(methods, attack)
    if "textcolor{red}" in "\n".join(body):
        caption += (r" Red entries are provisional (runs still completing); "
                    r"a dash ($-$) marks a configuration not yet evaluated.")
    # footnotesize, with column separation tuned per table. At the original
    # footnotesize/4pt this 13-column grid ran past the physical page edge (615pt on a
    # 612pt page), clipping the last Edge-IIoTset column, and LaTeX reported no overfull
    # box for it. Separation, not font size, is what sets the width here, so each table
    # gets the value that fills the 514pt text block without crossing 563pt: the widest
    # content (sign-flip) needs 1pt, the narrower label-flip and ablation grids can
    # afford more.
    L = [r"\begin{table*}[!t]\centering\footnotesize",
         r"\caption{" + title + r"}",
         r"\label{" + label + r"}", r"\setlength{\tabcolsep}{" + colsep + r"}",
         r"\begin{tabular}{@{}l" + "ccc" * 4 + r"@{}}\toprule",
         r"& \multicolumn{3}{c}{CIFAR-10} & \multicolumn{3}{c}{FashionMNIST} & \multicolumn{3}{c}{EMNIST} & \multicolumn{3}{c}{Edge-IIoTset}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}",
         r"Method & " + " & ".join([f"$f{{=}}.{int(f*10)}$" for _ in DSROOT for f in FS]) + r"\\\midrule"]
    L += body
    L += [r"\bottomrule\end{tabular}", r"\tablenote{" + caption + r"}", r"\end{table*}"]
    open(os.path.join(TBL, outfile), "w").write("\n".join(L))
    return prev


def _signflip_body(methods, attack):
    """The data rows, built before the caption so the legend can reflect them."""
    prev = ["METHOD".ljust(30) + "".join(f"{ds[:8]:>26}" for ds in DSROOT)]
    best = {}
    for ds, dsdir in DSROOT.items():
        for f in FS:
            vals = {a: val(runs(dsdir, a, attack, f)) for a, _ in methods}
            best[(ds, f)] = max((v[0], a) for a, v in vals.items() if v)[1] if any(vals.values()) else None
    body = []
    for agg, lbl in methods:
        cells, pcells = [], []
        for ds, dsdir in DSROOT.items():
            for f in FS:
                v = val(runs(dsdir, agg, attack, f))
                cells.append(fmt(v, bold=(best.get((ds, f)) == agg)))
                pcells.append(f"{v[0]:.0f}" if v else "--")
        body.append(f"{lbl} & " + " & ".join(cells) + r"\\")
        prev.append(lbl.ljust(30) + "".join(f"{c:>9}" for c in pcells))
    return body, prev


# TABLE I (main): baselines + FedGT + our flagship (hybrid) only.
main_methods = [("mean", "FedAvg (mean)~\\cite{mcmahan2017}"), ("trimmed", "Trimmed mean~\\cite{yin2018}"),
                ("median", "Median~\\cite{yin2018}"), ("krum", "Multi-Krum~\\cite{blanchard2017}"),
                ("bulyan", "Bulyan~\\cite{bulyan}"), ("geomedian", "Geo-median / RFA~\\cite{rfa}"),
                ("fltrust", "FLTrust~\\cite{fltrust}"), ("fedgt", "FedGT~\\cite{fedgt}"),
                ("hybrid_ov4", "\\textbf{CB-SAFE+ (ours)}")]
main_title = r"Test Accuracy (\%) Under the Sign-Flip Attack"
main_cap = (r"Four datasets and malicious fractions $f$ (mean\,$\pm$\,std over 3 seeds, 50 rounds). "
            r"Higher is better; best per column in bold. "
            r"CB-SAFE+ variants are ablated in Table~\ref{tab:ablation-variants}.")
prev = build_signflip_table(main_methods, main_title, main_cap, "tab:signflip", "table1_signflip.tex")
print("=== TABLE I: sign-flip robustness, main (preview, acc%) ===")
print("\n".join(prev))

# TABLE Ib (ablation): CB-SAFE+ components + trust-free capability.
abl_methods = [("reputation", "Base: temporal reputation (ov1)"),
               ("reputation_ov4", "\\;+ overlapping groups (ov4)"),
               ("hybrid_ov4", "\\;+ hybrid COMP decode (full)"),
               ("reputation_tf", "Trust-free (no root data)")]
abl_title = r"Ablation of CB-SAFE+ Components Under Sign-Flip"
abl_cap = (r"Same protocol as Table~\ref{tab:signflip}. The trust-free variant uses no "
           r"root dataset. "
           r"Best per column in bold.")
prevA = build_signflip_table(abl_methods, abl_title, abl_cap, "tab:ablation-variants",
                             "table2_ablation.tex", colsep="1.5pt")
print("\n=== TABLE Ib: CB-SAFE+ ablation (preview, acc%) ===")
print("\n".join(prevA))

# TABLE (secondary): label-flip accuracy, same roster as the main table.
lf_title = r"Test Accuracy (\%) Under the Label-Flip Attack"
lf_cap = (r"Four datasets and malicious fractions $f$ (mean\,$\pm$\,std over 3 seeds, "
          r"50 rounds). "
          r"Higher is better; best per column in bold.")
prevLF = build_signflip_table(main_methods, lf_title, lf_cap, "tab:labelflip", "table3_labelflip.tex",
                              attack="labelflip", colsep="2.5pt")
print("\n=== TABLE (secondary): label-flip (preview, acc%) ===")
print("\n".join(prevLF))


# ============ TABLE: no-attack (f=0) reference, one column per dataset ============
def build_f0_table(methods, outfile="table_f0.tex"):
    """An f=0 row for every rule, so attack damage can be told apart from the cost
    of each rule's own aggregation. Reads the robust_none_*_f00_* runs."""
    dsets = list(DSROOT.items())
    rows, allcells, prev = [], [], []
    for agg, lbl in methods:
        cells = []
        for _ds, dsdir in dsets:
            # FedGT: at f=0 there is no attack, so the BCJR decoder is inactive and
            # the run reduces to its aggregation path. The faithful-decoder directory
            # has no f=0 runs; read the same harness the other rules use, which does.
            paths = (glob.glob(os.path.join(R, dsdir, "robust_none_fedgt_f00_c3_s*.csv")
                               if dsdir else
                               os.path.join(R, "robust_none_fedgt_f00_c3_s*.csv"))
                     if agg == "fedgt" else runs(dsdir, agg, "none", 0.0))
            cells.append(fmt(val(paths)))
        rows.append(f"{lbl} & " + " & ".join(cells) + r"\\")
        allcells += cells
        prev.append(lbl.ljust(34) + "".join(f"{c[:18]:>20}" for c in cells))
    cap = (r"Accuracy at $f{=}0$ across all four benchmarks "
           r"(mean\,$\pm$\,std over 3 seeds, 50 rounds). Every rule is applied to benign "
           r"updates only. FedGT uses the same harness as the other rules here, its "
           r"group-testing decoder being inactive without an attack."
           + _legend_if_needed(allcells))
    L = [r"\begin{table}[!t]\centering\footnotesize",
         r"\caption{No-Attack Reference Accuracy (\%)}", r"\label{tab:noattack}",
         r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{@{}l" + "c" * len(dsets) + r"@{}}\toprule",
         "Method & " + " & ".join(ds for ds, _ in dsets) + r"\\\midrule"] + rows
    L += [r"\bottomrule\end{tabular}}", r"\tablenote{" + cap + r"}", r"\end{table}"]
    open(os.path.join(TBL, outfile), "w").write("\n".join(L))
    return prev


prevF0 = build_f0_table(main_methods)
print("\n=== TABLE: no-attack f=0 reference (preview, acc%) ===")
print("\n".join(prevF0))


# ============ TABLE II: FedGT head-to-head + identification + cost (CIFAR + FMNIST) ============
methods2 = [("fedgt", "FedGT [TIFS'25]"), ("reputation", "CB-SAFE+ ov1"),
            ("reputation_ov4", "CB-SAFE+ ov4"), ("reputation_tf", "CB-SAFE+ trust-free"),
            ("hybrid_ov4", "CB-SAFE+ hybrid")]
cost = {"fedgt": "40", "reputation": "10", "reputation_ov4": "40",
        "reputation_tf": "10", "hybrid_ov4": "40"}
rootfree = {"fedgt": "no", "reputation": "no", "reputation_ov4": "no",
            "reputation_tf": "\\textbf{yes}", "hybrid_ov4": "no"}
print("\n=== TABLE II: FedGT head-to-head (CIFAR acc | FMNIST acc | ident caught | FP | cost | root-free) ===")
print("METHOD".ljust(22) + "cifar f.1/.2/.3    fmnist f.1/.2/.3    caught(cifar)   FP        ops  rootfree")
for agg, label in methods2:
    ca = " ".join(f"{(val(runs('',agg,'signflip',f)) or (0,0,0))[0]:.0f}" if val(runs('',agg,'signflip',f)) else "--" for f in FS)
    fa = " ".join(f"{(val(runs('fmnist',agg,'signflip',f)) or (0,0,0))[0]:.0f}" if val(runs('fmnist',agg,'signflip',f)) else "--" for f in FS)
    cg = " ".join(ident('', agg, f) for f in FS)
    fp = " ".join(identfp('', agg, f) for f in FS)
    print(f"{label:22}{ca:12}   {fa:12}   {cg:14}{fp:10}{cost[agg]:5}{rootfree[agg]}")

# ============ TABLE III: crypto + overhead ============
k = pd.read_csv(os.path.join(R, "crypto_kem_bench.csv")).set_index("name")
o = pd.read_csv(os.path.join(R, "overhead.csv"))
o = o[o.cluster_size == 3].set_index("kem")
print("\n=== TABLE III: crypto primitives + per-client overhead (c=3) ===")
print("KEM         fam    lvl   pk    ct   setupUp setupDn wall_s  roundKiB mask_ms unmask_s")
order = ["hqc-128", "hqc-192", "hqc-256", "mlkem-512", "mlkem-768", "mlkem-1024"]
for kem in order:
    kr = k.loc[kem]; orr = o.loc[kem]
    print(f"{kem:11} {kr['family'][:4]:5} {kr['claimed_nist_level']:>3}  {kr['public_key_bytes']:>5} {kr['ciphertext_bytes']:>5}"
          f"  {orr['setup_up_B']:>6} {orr['setup_down_B']:>6} {orr['setup_wall_s']:>6.2f}  {(orr['round_up_B']+orr['round_down_B'])/1024:>7.0f}"
          f"  {orr['round_client_mask_s']*1000:>5.0f}  {orr['round_server_unmask_s']:>6.2f}")

# ============ TABLE IV: backdoor ASR + label-flip acc ============
print("\n=== TABLE IV: backdoor ASR (lower better) + label-flip acc, by dataset/method/f ===")
print("dataset   method       BD-ASR f.1/.2/.3     LF-acc f.1/.2/.3")
for ds, dsdir in DSROOT.items():
    for agg, label in [("mean", "mean"), ("median", "median"), ("krum", "Krum"), ("reputation", "CB-SAFE+")]:
        bd = " ".join(f"{v[0]:.0f}" if (v:=val(runs(dsdir,agg,'backdoor',f),asr=True)) else "--" for f in FS)
        lf = " ".join(f"{v[0]:.0f}" if (v:=val(runs(dsdir,agg,'labelflip',f))) else "--" for f in FS)
        if bd.strip("- ") or lf.strip("- "):
            print(f"{ds[:9]:9} {label:12} {bd:18} {lf}")

print("\nwrote LaTeX table1_signflip.tex (+ others buildable); table*/full-width, packed.")
