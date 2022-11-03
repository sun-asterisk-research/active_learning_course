import torch

from .base_strategy import BaseStrategy


class BALDSampling(BaseStrategy):
    def __init__(self, dataset, net):
        super(BALDSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_bayesian_split(unlabeled_data)
        mean_prob = probs.mean(0)
        entropy1 = (-mean_prob * torch.log(mean_prob)).sum(1)
        entropy2 = (-probs * torch.log(probs)).sum(2).mean(0)

        uncertainties = entropy2 - entropy1
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
