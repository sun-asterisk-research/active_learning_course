from .base_strategy import BaseStrategy


class LeastConfidenceSampling(BaseStrategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
