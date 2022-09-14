from dataloader import get_mnist
from hyperparams import BATCH_SIZE, LEARNING_RATE, MOMENTUM, N_EPOCHS, N_WORKERS
from models import ActiveNet, MNISTNet
from sampling_strategies import (
    EntropySampling,
    LeastConfidenceSampling,
    MarginSampling,
    RandomSampling,
    RatioSampling,
)

params = {
    "MNIST": {
        "n_epoch": N_EPOCHS,
        "train_args": {"batch_size": BATCH_SIZE, "num_workers": N_WORKERS},
        "test_args": {"batch_size": 1024, "num_workers": N_WORKERS},
        "optimizer_args": {"lr": LEARNING_RATE, "momentum": MOMENTUM},
    }
}


def get_dataset(name):
    if name == "MNIST":
        return get_mnist()
    else:
        raise NotImplementedError


def get_net(name, device):
    if name == "MNIST":
        return ActiveNet(MNISTNet, params[name], device)
    else:
        raise NotImplementedError


def get_params(name):
    return params[name]


def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "RatioSampling":
        return RatioSampling
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "LeastConfidenceSampling":
        return LeastConfidenceSampling
    else:
        raise NotImplementedError
