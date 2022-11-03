import numpy as np
from sklearn.metrics import pairwise_distances

from .base_strategy import BaseStrategy


class KCentersGreedySampling(BaseStrategy):
    def __init__(self, dataset, net):
        super(KCentersGreedySampling, self).__init__(dataset, net)

    def greedy_select(self, X, X_set, n):
        m = np.shape(X)[0]
        dist_ctr = pairwise_distances(X, X_set)
        min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        # Get embeddings
        labeled_embeddings = self.get_embeddings(labeled_data)
        unlabeled_embeddings = self.get_embeddings(unlabeled_data)

        # Chosen with greedy strategy
        chosen = self.greedy_select(unlabeled_embeddings, labeled_embeddings, n)

        return unlabeled_idxs[chosen]
