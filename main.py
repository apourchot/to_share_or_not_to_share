import os
from argparse import ArgumentParser

import torch
import yaml
from torch.utils.data.dataset import Subset
from torchvision import datasets

from models.factories import graph_sampler_factory, model_factory
from training import eval_supernet, train_supernet
from utils import TFRMSprop, data_transforms_cifar10, get_output_folder


def get_scheduler(optimizer, config):
    """
    Creates the scheduler from the config
    """
    if config["name"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["T_max"])
        return scheduler
    else:
        raise NotImplementedError


def get_optimizer(params, config):
    """
    Creates an optimizer from the given config
    """
    if config["name"] == "SGD":
        optimizer = torch.optim.SGD(params, lr=config["lr"], momentum=config["momentum"],
                                    weight_decay=config["weight_decay"])
        return optimizer
    if config["name"] == "TFRMSprop":
        optimizer = TFRMSprop(params, lr=config["lr"], momentum=config["momentum"],
                              eps=config["eps"], weight_decay=config["weight_decay"])
        return optimizer
    else:
        raise NotImplementedError


def get_criterion(config):
    """
    Creates the torch criterion for optimization
    """
    if config["name"] == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError


def get_data_iters(config):
    """
    Creates the train and validation data iterators on the given dataset
    :param config:
    :return:
    """
    if config["name"] == "cifar10":

        # Loading Data
        train_transform, valid_transform = data_transforms_cifar10(cutout=False, cutout_length=16)
        data_train = datasets.CIFAR10("./datasets/cifar10", train=True, download=True, transform=train_transform)
        data_valid = datasets.CIFAR10("./datasets/cifar10", train=False, download=True, transform=valid_transform)

        # building cyclic iterators over the training and validation sets
        loader_train = torch.utils.data.DataLoader(data_train, batch_size=int(config["batch_size"]), shuffle=True)
        loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=int(config["batch_size"]), shuffle=True)

        print("Length of datasets: Train: {}, Valid: {}".format(len(data_train), len(data_valid)))
        print("Length of loaders: Train: {}, Valid: {}".format(len(loader_train), len(loader_valid)))

        return loader_train, loader_valid

    elif config["name"] == "cifar10-valid":

        # Loading Data
        train_transform, valid_transform = data_transforms_cifar10(cutout=False, cutout_length=16)
        dataset_train = datasets.CIFAR10("./datasets", train=True, download=True, transform=train_transform)
        dataset_valid = datasets.CIFAR10("./datasets", train=True, download=True, transform=valid_transform)

        # training set contains 40,000 images, validation and test set contain 10,000 images
        dataset_valid = Subset(dataset_valid, range(4 * len(dataset_train) // 5, len(dataset_train)))
        dataset_train = Subset(dataset_train, range(4 * len(dataset_train) // 5))

        # building cyclic iterators over the training and validation sets
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
        loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=config["batch_size"], shuffle=True)

        print("Length of datasets: Train: {}, Valid: {}".format(len(dataset_train), len(dataset_valid)))
        print("Length of loaders: Train: {}, Valid: {}".format(len(loader_train), len(loader_valid)))

    else:
        raise NotImplementedError


if __name__ == "__main__":

    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--snapshot", type=str, default=None, required=False)
    parser.add_argument("--train_only", dest="train_only", action="store_true")
    parser.add_argument("--eval_only", dest="eval_only", action="store_true")
    args = parser.parse_args()

    # cuda device
    device = torch.cuda.device(0)

    # load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("Config:", config)

    # creating the super-net and the sampler
    model = model_factory(config["TRAIN"]["GRAPH_MODEL"])
    if args.snapshot is not None:
        model.load_state_dict(torch.load(args.snapshot_path, map_location=device))
    print("Number of parameters: {}".format(sum([p.numel() for p in model.parameters()])))
    task_sampler = graph_sampler_factory(config["TRAIN"]["GRAPH_SAMPLER"])

    # training stuff
    optimizer = get_optimizer(model.parameters(), config["TRAIN"]["OPTIMIZER"])
    scheduler = get_scheduler(optimizer, config["TRAIN"]["SCHEDULER"])
    criterion = get_criterion(config["TRAIN"]["CRITERION"])

    # data iterators
    loader_train, loader_valid = get_data_iters(config["TRAIN"]["DATA"])

    # device
    device = torch.cuda.current_device()
    model.to(device)

    # folder where results are saved
    model_dir = get_output_folder("./results", config["TRAIN"]["exp_name"])

    # If training, skipped if only evaluating
    if not args.eval_only:
        train_supernet(model_dir, model, graph_s=task_sampler,
                       criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                       train_iter=loader_train, valid_iter=loader_valid,
                       config=config["TRAIN"], device=device)

    # If evaluating, skipped if only training
    if not args.train_only:
        eval_supernet(model_dir, model, task_sampler=task_sampler,
                      train_iter=loader_train, valid_iter=loader_valid,
                      config=config["EVAL"], device=device)
