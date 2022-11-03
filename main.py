import os

import torch

# Apply WanDB
import wandb
from hyperparams import *
from utils import get_dataset, get_net, get_strategy

# Reconfig your WANDB API Key here
os.environ["WANDB_API_KEY"] = WANDB_KEY
os.environ["WANDB_BASE_URL"] = WANDB_HOST

# Login wandb
wandb.login()

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dataset_name = "MNIST"

# Lesson 1
strategy_names = [
    # "RandomSampling",
    # "LeastConfidenceSampling",
    # "MarginSampling",
    # "EntropySampling",
    # "RatioSampling",
    # "BNNSampling",
    # "MCDropoutSampling",
    "BALDSampling",
]


def train_active_learning(dataset, net, strategy):
    # start experiment
    dataset.initialize_labels(N_INIT_LABELED)

    # Log to WANDB
    wandb.init(
        project=WANDB_PROJECT,
        name=f"{dataset_name} - {strategy_name}",
        config={
            "batch_size": BATCH_SIZE,
            "n_init_labeled": N_INIT_LABELED,
            "n_query": N_QUERY,
            "n_epochs": N_EPOCHS,
            "n_round": N_ROUND,
            "lr": LEARNING_RATE,
            "momentum": MOMENTUM,
        },
    )
    wandb.define_metric("acc")
    wandb.define_metric("round")

    # round 0 accuracy
    print("Round 0 training...")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")

    wandb.log({"round": 0, "acc": dataset.cal_test_acc(preds)})

    # Iterative learning
    for rd in range(1, N_ROUND + 1):
        print(f"Round {rd} training...")

        # query
        query_idxs = strategy.query(N_QUERY)

        # update labels
        strategy.update(query_idxs)
        strategy.train()

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        acc = dataset.cal_test_acc(preds)

        print(f"Round {rd} testing accuracy: {acc}")

        wandb.log({"round": rd, "acc": acc})

    # Call wandb.finish() when end of experiment
    wandb.finish()


# Run experiments
for strategy_name in strategy_names:
    print(f"RUNNING STRATEGY {strategy_name}")
    # Load dataset
    dataset = get_dataset(dataset_name)
    # Load network
    torch.manual_seed(SEED)
    net = get_net("MNIST", device)
    # Load strategy
    strategy = get_strategy(strategy_name)(dataset, net)
    train_active_learning(dataset, net, strategy)
