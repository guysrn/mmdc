import random
import pickle

import numpy as np
from scipy.spatial.distance import cdist


R = 0.1


class GaussianMixtureModel:
    """
    A Gaussian mixture model distribution. Assuming covariance matrix is sigma*I (Identity).
    """

    def __init__(self, sigma, k, dim, normalize=True, init="one_hot"):
        self.sigma = sigma
        self.k = k
        self.dim = dim
        self.normalize = normalize
        self.means = self._init_means(init)

    def _init_means(self, init):
        if init == "random":
            means = np.random.uniform(-R, R, size=(self.k, self.dim))
        elif init == "one_hot":
            means = np.eye(self.k)
        else:
            raise ValueError(f"Unknown means init option: {init}")

        if self.normalize:
            for i, mean in enumerate(means):
                means[i] = mean / np.linalg.norm(mean, 2)

        return means

    def sample(self, n):
        ret = np.zeros((n, self.dim))
        for i in range(n):
            s = np.random.normal(self.means[random.randint(0, self.k - 1), :], self.sigma, self.dim)
            if self.normalize:
                s = s / np.linalg.norm(s, 2)
            ret[i] = s
        return ret

    def get_labels(self, samples, one_hot=False):
        labels = np.argmin(cdist(self.means, samples), axis=0)
        if one_hot:
            return np.eye(self.k)[labels]
        return labels

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump((self.sigma, self.k, self.dim, self.means), handle)

    def load(self, path):
        with open(path, "rb") as handle:
            self.sigma, self.k, self.dim, self.means = pickle.load(handle)
