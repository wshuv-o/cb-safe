import _bootstrap  # noqa: F401
import csv
import os

import numpy as np

D100 = os.path.join(_bootstrap.RESULTS, "scale")
D500 = os.path.join(_bootstrap.RESULTS, "scale50")
SEEDS_100 = [0, 1, 2]
SEEDS_500 = [0, 3, 2]
N_HON, N_MAL = 400, 100

ROWS = [
    ("hybrid_ov4", r"\textbf{CB-SAFE+ (ours)}", True),
    # FedGT omitted at N=500: its parity-check matrices / prevalence tables are
    # hardcoded for n<=31 (Xhemrishi et al.), so it has no configuration at this scale.
    ("fltrust",    r"FLTrust~\cite{fltrust}",    False),
    ("median",     r"Median",                    False),
    ("geomedian",  r"Geo-median~\cite{rfa}",     False),
    ("krum",       r"Multi-Krum",                False),
    ("bulyan",     r"Bulyan~\cite{bulyan}",      False),
    ("trimmed",    r"Trimmed mean",              False),
    ("mean",       r"Mean",                      False),
]


# Every rule is read over the same horizon: the CB-SAFE+ N=500 runs continue to
# 60 rounds, so they are truncated to the 50 rounds the baselines ran.
CAP = {D100: 30, D500: 50}


def last5(path, cap):
    if not os.path.exists(path):
        return None
    r = list(csv.DictReader(open(path)))
    if len(r) < cap:
        return None
    r = r[:cap]
    return np.mean([float(x["acc"]) for x in r[-5:]]) * 100


def acc_vals(dirpath, stem, seeds):
    nstr = "100" if dirpath == D100 else "500"
    return [last5(os.path.join(dirpath, f"robust_signflip_{stem}_N{nstr}_f20_c3_s{s}.csv"),
                  CAP[dirpath]) for s in seeds]


def det_vals(stem, ident):
    if not ident:
        return None
    caught, fps = [], []
    for s in SEEDS_500:
        p = os.path.join(D500, f"robust_signflip_{stem}_N500_f20_c3_s{s}.csv")
        if not os.path.exists(p):
            return "run"
        r = list(csv.DictReader(open(p)))
        if len(r) < 50:
            return "run"
        r = r[:50]
        caught.append(int(r[-1]["excluded_malicious"]))
        fps.append(int(r[-1]["excluded_honest"]))
    c, f = np.mean(caught), np.mean(fps)
    return dict(caught=c, fp=f, pfa=f / N_HON, pmd=(N_MAL - c) / N_MAL)


def meanstd(vals):
    got = [v for v in vals if v is not None]
    if len(got) == len(vals) and got:
        return np.mean(got), np.std(got, ddof=1)
    return None


rows = []
for stem, name, ident in ROWS:
    a100 = acc_vals(D100, stem, SEEDS_100)
    a500 = acc_vals(D500, stem, SEEDS_500)
    rows.append(dict(name=name, ident=ident, a100=a100, a500=a500,
                     m100=meanstd(a100), m500=meanstd(a500), det=det_vals(stem, ident)))


def col_best(key, i, fn=max):
    vs = [r[key][i] for r in rows if r[key][i] is not None]
    return fn(vs) if vs else None


best_a100 = [col_best("a100", i) for i in range(3)]
best_a500 = [col_best("a500", i) for i in range(3)]
best_m100 = max([r["m100"][0] for r in rows if r["m100"]], default=None)
best_m500 = max([r["m500"][0] for r in rows if r["m500"]], default=None)
dets = [r["det"] for r in rows if isinstance(r["det"], dict)]
best_caught = max([d["caught"] for d in dets], default=None)
best_fp = min([d["fp"] for d in dets], default=None)
best_pfa = min([d["pfa"] for d in dets], default=None)
best_pmd = min([d["pmd"] for d in dets], default=None)


def bold(s, cond):
    return f"\\textbf{{{s}}}" if cond else s


def eq(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-9


def fmt(v, best, prec=1):
    if v is None:
        return "--"
    return bold(f"{v:.{prec}f}", eq(v, best))


def fmt_mean(ms, best):
    if ms is None:
        return r"\textit{run}"
    m, sd = ms
    return bold(f"{m:.1f}$\\pm${sd:.1f}", eq(m, best))


lines = []
for r in rows:
    a100 = [fmt(r["a100"][i], best_a100[i]) for i in range(3)]
    a500 = [fmt(r["a500"][i], best_a500[i]) for i in range(3)]
    d = r["det"]
    if d is None:
        det = ["--", "--", "--", "--", r"$\times$"]
    elif d == "run":
        det = [r"\textit{run}"] * 4 + [r"\checkmark"]
    else:
        det = [bold(f"{d['caught']:.0f}/{N_MAL}", eq(d["caught"], best_caught)),
               fmt(d["fp"], best_fp, 0), fmt(d["pfa"], best_pfa, 3),
               fmt(d["pmd"], best_pmd, 3), r"\checkmark"]
    cells = [r["name"]] + a100 + [fmt_mean(r["m100"], best_m100)] + \
            a500 + [fmt_mean(r["m500"], best_m500)] + det
    lines.append(" & ".join(cells) + r" \\")

table = r"""\begin{table*}[!t]
\caption{Scaling with the Client Population}
\label{tab:scale-n500}
\centering
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l cccc cccc ccccc@{}}
\toprule
& \multicolumn{4}{c}{$N{=}100$ (30 rd)} & \multicolumn{4}{c}{$N{=}500$ (50 rd)} & \multicolumn{5}{c}{Detection ($N{=}500$)}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-14}
Method & s0 & s1 & s2 & mean$\pm$sd & s0 & s1 & s2 & mean$\pm$sd & Caught & FP$\downarrow$ & $P_{\mathrm{FA}}\downarrow$ & $P_{\mathrm{MD}}\downarrow$ & Id.\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\tablenote{Test accuracy (\%) at $N{=}100$ (30
rounds) and $N{=}500$ (50 rounds) under sign-flip $f{=}0.2$ on FashionMNIST
(Dirichlet $\alpha{=}0.5$); per-seed values are the mean of the final five rounds,
over three seeds. The detection block reports, at $N{=}500$, malicious clients
caught of 100, honest clients wrongly excluded (FP) of 400, and the derived
false-alarm ($P_{\mathrm{FA}}$) and missed-detection ($P_{\mathrm{MD}}$) rates. Best
value per column in bold.}
\end{table*}"""

out = os.path.join(_bootstrap.RESULTS, "tables", "table_n500_scale.tex")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write(table + "\n")
print(table)
print("\n-- written to", os.path.normpath(out))
