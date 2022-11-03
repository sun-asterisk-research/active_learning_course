import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbnn as bnn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ActiveNet:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def train(self, data):
        n_epoch = self.params["n_epoch"]
        self.clf = self.net().to(self.device)
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params["optimizer_args"])

        loader = DataLoader(data, shuffle=True, **self.params["train_args"])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_bayesian(self, data, n_infer=10, mc_dropout=False):
        self.clf.eval()
        if mc_dropout:
            # Use dropout in inference
            self.clf.train()

        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        for _ in range(n_infer):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_infer
        return probs

    def predict_prob_bayesian_split(self, data, n_infer=10):
        self.clf.train()

        probs = torch.zeros([n_infer, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        for i in range(n_infer):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += prob.cpu().data
        return probs

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params["test_args"])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        el = F.relu(self.fc1(x))
        x = self.fc1_drop(el)
        x = self.fc2(x)
        return x, el

    def get_embedding_dim(self):
        return 128


class MNISTBayesianNet(nn.Module):
    def __init__(self):
        super(MNISTBayesianNet, self).__init__()
        self.fc1 = bnn.BayesLinear(
            prior_mu=0, prior_sigma=0.1, in_features=28 * 28, out_features=128
        )
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = bnn.BayesLinear(
            prior_mu=0, prior_sigma=0.1, in_features=128, out_features=10
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        el = F.relu(self.fc1(x))
        x = self.fc1_drop(el)
        x = self.fc2(x)
        return x, el
