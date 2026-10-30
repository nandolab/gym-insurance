import os
import random
from math import log10
from matplotlib import pyplot as plt

import benfordslaw.benfordslaw as bf
import numpy as np
import pandas as pd
from benfordslaw.benfordslaw import _count_digit
from scipy.spatial import distance

decisions = {1: "approved", 0: "rejected", "decision": "all"}


class Benford(object):
    def __init__(self, d=2) -> None:
        self.d = d
        self.drange = range(1, d + 1)
        self.bfs = [bf(pos=d, verbose=0, alpha=0.01) for d in self.drange]

    def update_drange(self, d):
        self.drange = range(1, d + 1)

    def _fit(self, x):
        return [self.bfs[d - 1].fit(x) for d in self.drange]

    def _vec_dist(self, x):
        emp_dist = self._fit(x)
        return [emp_dist[d - 1]["percentage_emp"][:, 1] for d in self.drange]

    def distance(self, x):
        dists = 0
        emp_dist = self._vec_dist(x)
        _dist = []
        for d in self.drange:
            len(self.bfs[d - 1].leading_digits) == emp_dist[d - 1]
            _dist.append(
                np.max(self.bfs[d - 1].leading_digits - emp_dist[d - 1])
                / np.linalg.norm(self.bfs[d - 1].leading_digits - emp_dist[d - 1])
            )
        _dist_mean = np.mean(_dist)
        if np.isnan(_dist_mean):
            return 1.0
        else:
            return _dist_mean

    def chi_test(self, x):
        tests = self._fit(x)
        return [tests[d - 1]["P_significant"] for d in self.drange]

    def plot(self, x):
        self._fit(x)
        return [self.bfs[d - 1].plot(title=f"d={d}")[0] for d in self.drange]

    def save_plot(self, x, path, decision=None):
        self._fit(x)
        for d in self.drange:
            os.makedirs(path, exist_ok=True)
            file_path = path / f"{decisions[decision]}_DIGIT_{d}.png"
            _plot = self.bfs[d - 1].plot(
                title=f"""Distribution of d={d} and n:{len(x)} for Decision: {decision}"""
            )[0]
            plt.close()
            _plot.savefig(file_path)


class BenfordRandom:
    def __init__(self):
        self.first_digit_distribution = [log10(1 + 1 / i) for i in range(1, 10)]

    def rand(self):
        rand_p = random.random()
        for i in range(1, 10):
            if rand_p < self.first_digit_distribution[i - 1]:
                return i
        return 9

    def randint(self, a, b, size=None):
        if size is None:
            return self.rand() * (10 ** (int(log10(b)) - 1)) + random.randint(a, b)
        else:
            return np.array(
                [
                    self.rand() * (10 ** (int(log10(b)) - 1)) + random.randint(a, b)
                    for _ in range(size)
                ]
            )
