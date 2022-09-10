from dataloader import get_mnist
from models import MNISTNet, ActiveNet
from sampling_strategies import MarginSampling, EntropySampling, RandomSampling, LeastConfidenceSampling, RatioSampling

params = {
    'MNIST': {
        'n_epoch': 10,
        'train_args': {'batch_size': 64, 'num_workers': 1},
        'test_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5}
    }
}


def get_dataset(name):
    if name == 'MNIST':
        return get_mnist()
    else:
        raise NotImplementedError


def get_net(name, device):
    if name == 'MNIST':
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
