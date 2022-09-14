import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from hyperparams import SEED


class MNISTDataLoader:
    def __init__(self, X_train, Y_train, X_test, Y_test, dataset):
        np.random.seed(SEED)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.dataset = dataset

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.dataset(
            self.X_train[labeled_idxs], self.Y_train[labeled_idxs]
        )

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.dataset(
            self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]
        )

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.dataset(self.X_train, self.Y_train)

    def get_test_data(self):
        return self.dataset(self.X_test, self.Y_test)

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test


class MNISTDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x.numpy(), mode="L")
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


def get_mnist():
    raw_train = datasets.MNIST("./data/MNIST", train=True, download=True)
    raw_test = datasets.MNIST("./data/MNIST", train=False, download=True)
    return MNISTDataLoader(
        raw_train.data[:40000],
        raw_train.targets[:40000],
        raw_test.data[:40000],
        raw_test.targets[:40000],
        MNISTDataset,
    )
