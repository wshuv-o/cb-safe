"""Path + environment bootstrap for experiment scripts. Import this first."""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OQS_INSTALL_PATH", os.path.expanduser("~/_oqs"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

RESULTS = os.environ.get("CBSAFE_OUT") or os.path.join(REPO, "results")
try:
    os.makedirs(RESULTS, exist_ok=True)
except OSError:
    # repo is on a read-only mount (e.g. Kaggle /kaggle/input) - use a writable dir
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    RESULTS = os.path.join(base, "results")
    os.makedirs(RESULTS, exist_ok=True)
