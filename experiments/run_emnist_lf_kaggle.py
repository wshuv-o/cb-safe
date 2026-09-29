"""EMNIST label-flip Kaggle runner (all 12 methods, 1 seed, 30 rounds).

Half of the label-flip sweep runs here (EMNIST); the two image sets CIFAR-10 and
FashionMNIST run locally. EMNIST auto-downloads via torchvision, so NO dataset
needs to be added to the notebook. Rounds are forced to 30 to match the local
EMNIST runs. Writes /kaggle/working/results/kaggle/emnist/robust_labelflip_*.csv.
Resumable: re-run to continue; finished CSVs are skipped.

Enable a T4 GPU + Internet, then run this.
"""

import os

os.environ["CBSAFE_DATASETS"] = "emnist"
os.environ["CBSAFE_ATTACKS"] = "labelflip"
os.environ["CBSAFE_ROUNDS"] = "30"

import run_all_kaggle  # noqa: E402


if __name__ == "__main__":
    run_all_kaggle.main()
