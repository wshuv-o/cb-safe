from __future__ import annotations

import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLL = os.path.join(_HERE, "BCJR_4_python.dll")
_H30 = os.path.join(_HERE, "fedgt_H30.npy")

# FedGT's #-negative-tests -> estimated-#-malicious lookup (their group_test.py, n=30)
LOOK_UP_NM_30 = [8, 8, 8, 8, 7, 6, 5, 5, 4, 3, 2, 1, 0]


class GroupTestFaithful:
    def __init__(self, n_clients: int = 30, P_MD_test: float = 0.05, P_FA_test: float = 0.05):
        assert n_clients == 30, "faithful FedGT is defined only for its published n=30 config"
        self.parity_check_matrix = np.load(_H30).astype(np.uint8)
        self.n_tests, self.n_clients = self.parity_check_matrix.shape
        assert (self.n_tests, self.n_clients) == (12, 30)

        lib = ctypes.CDLL(_DLL)
        self.fun = lib.BCJR
        self.fun.restype = None
        p_ui8 = ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")
        p_d = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
        self.fun.argtypes = [p_ui8, p_d, p_ui8, p_d, ctypes.c_double,
                             ctypes.c_int, ctypes.c_int, p_d, p_ui8]
        self.LLRO = np.empty((1, self.n_clients), dtype=np.double)
        self.DEC = np.empty((1, self.n_clients), dtype=np.uint8)
        self.ChannelMatrix = np.array([[1 - P_FA_test, P_FA_test],
                                       [P_MD_test, 1 - P_MD_test]], dtype=np.double)

    # ---- verbatim from FedGT/defence/group_test.py ----
    def _dunn_index(self, data, labels, centroids):
        num_samples = len(data)
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        if num_clusters == 1:
            return 0
        dist_centr = np.inf * np.ones((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                dist_centr[i, j] = dist_centr[j, i] = np.linalg.norm(centroids[i] - centroids[j])
        distances = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                distances[i, j] = distances[j, i] = np.linalg.norm(data[i] - data[j])
        a = np.zeros(num_samples)
        denumerator = 0
        for i in range(num_samples):
            same = np.where(labels == labels[i])[0]
            if len(same) > 1:
                a[i] = np.max(distances[i, same])
                if max(denumerator, a[i]) == a[i]:
                    denumerator = a[i]
        return dist_centr.min() / denumerator if denumerator else 0

    def perform_clustering_and_testing(self, group_acc, group_PCA, ss_thres=0.0):
        group_acc = np.asarray(group_acc, float)
        group_PCA = np.asarray(group_PCA, float)
        X = np.column_stack((group_acc, group_PCA))
        poss = np.arange(1, int(self.parity_check_matrix.sum(1).max()) + 2, dtype=int).tolist()
        s_scores = -np.ones(len(poss))
        d_scores = -np.ones(len(poss))
        all_labels = np.empty((len(poss), self.n_tests))
        for k, clst in enumerate(poss):
            km = KMeans(n_clusters=clst, n_init=10, random_state=0).fit(X)
            all_labels[k, :] = km.labels_
            # KMeans can collapse to <clst effective labels on degenerate (all-chance)
            # group features; silhouette_score then errors. Guard it (their code omits this).
            s_scores[k] = 0.0 if (clst == 1 or len(np.unique(km.labels_)) < 2) else silhouette_score(X, km.labels_)
            d_scores[k] = self._dunn_index(X, km.labels_, km.cluster_centers_)
        if s_scores.max() < ss_thres:
            return np.zeros(self.n_tests, dtype=np.uint8)
        best = int(np.argmax(d_scores))
        tot = poss[best]
        idx = {v: np.where(all_labels[best, :] == v)[0] for v in range(tot)}
        temp = -np.ones((tot, 3))
        for key, items in idx.items():
            temp[key, 0] = len(items)
            temp[key, 1] = np.mean(group_acc[items]) if items.size else -1
            temp[key, 2] = np.mean(group_PCA[items]) if items.size else -1
        temp = temp.T
        tests = np.ones(self.n_tests, dtype=np.uint8)
        if np.sum(temp[1, :] == np.max(temp[1, :])) == 1:
            tests[idx[int(np.argmax(temp[1, :]))]] = 0
            return tests
        sp = np.argsort(temp[2, :]); st = temp[:, sp]; si = np.arange(tot)[sp]
        if st[1, 0] > st[1, -1]:
            tests[idx[si[0]]] = 0
        elif st[1, 0] < st[1, -1]:
            tests[idx[si[-1]]] = 0
        else:
            if st[0, 0] > st[0, -1]:
                tests[idx[si[0]]] = 0
            elif st[0, 0] < st[0, -1]:
                tests[idx[si[-1]]] = 0
            else:
                tests = np.zeros(self.n_tests, dtype=np.uint8)
        return tests

    def perform_gt(self, test_values):
        tests = np.zeros((1, self.n_tests), dtype=np.uint8)
        tests[0, :] = test_values
        nm_est = LOOK_UP_NM_30[int(np.sum(tests[0, :] == 0))]
        prev_est = nm_est / self.n_clients
        if nm_est == 0:
            prev_est = 0.1
        LLRinput = np.log((1 - prev_est) / prev_est) * np.ones((1, self.n_clients), dtype=np.double)
        td = 0.0
        self.fun(self.parity_check_matrix, LLRinput, tests, self.ChannelMatrix, td,
                 self.n_clients, self.n_tests, self.LLRO, self.DEC)
        idx_sort = np.argsort(self.LLRO)
        if nm_est != 0:
            self.DEC[0, idx_sort[:, :nm_est]] = 1
            ident = np.where(self.LLRO[0, :] == self.LLRO[0, idx_sort][0, nm_est - 1])[0]
            if len(ident) > 1:
                self.DEC[0, ident] = 1
        return np.where(self.DEC[0] == 1)[0]      # flagged (defective) client indices
