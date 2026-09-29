"""Regenerate the tables, statistics, and figures from the result files.

    python analyze.py              # everything
    python analyze.py tables       # LaTeX tables -> results/tables/
    python analyze.py stats        # paired t-tests -> results/significance*.csv
    python analyze.py figures      # figures -> results/figs/

No training is involved; every output is computed from the CSVs in results/.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = {
    "tables": ["tables_main.py", "table_scaling.py"],
    "stats": ["stats.py"],
    "figures": ["fig_dynamics.py", "fig_accuracy_vs_f.py", "fig_gamma_sweep.py",
                "fig_suspicion.py", "fig_convergence_n500.py"],
}


def main() -> int:
    wanted = sys.argv[1:] or list(STEPS)
    unknown = [w for w in wanted if w not in STEPS]
    if unknown:
        print(f"unknown step(s): {', '.join(unknown)}; choose from {', '.join(STEPS)}")
        return 2
    status = 0
    for step in wanted:
        for script in STEPS[step]:
            print(f"== {step}: {script}", flush=True)
            rc = subprocess.call([sys.executable, os.path.join(HERE, "analysis", script)], cwd=HERE)
            status |= rc
    return status


if __name__ == "__main__":
    sys.exit(main())
