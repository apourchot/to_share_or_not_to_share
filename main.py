import os
from argparse import ArgumentParser

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets

from models.graph_comps import GraphComp
from models.graph_sampler import GraphSampler
from training import eval_supernet, train_supernet
from utils import TFRMSprop, data_transforms_cifar10, get_output_folder

exp_params = {

    # misc
    "device": torch.device(0),
    "num_epochs": 108 * 4,  # total number of epochs
    "perform_valid": False,  # evaluate on valid set after every epoch
    "use_tensorboard": True,
    "save": True,  # save models
    "save_period": 4,  # use higher values to save less models. Make sure that it divides the number of epochs
    "exp_dir": "",  # to load models
    "output_dir": "baseline_ft_bn",
    "train_seed": None,  # seed for training; will be generated and saved automatically if set to None
    "eval_seed": None,  # seed for evaluation; will be generated and saved automatically if set to None

    # eval stuff
    "n_rand_evals": 1000,  # number of models sampled and evaluated
    "ft_bn_stats": True,  # whether or not to ft only the bn stats before evaluating a model
    "n_ft_bn_stats": 4,  # number of mini-batches to compute bn stats on
    "ft_weights": False,  # whether or not to ft all the weights of the super-net to a specific model before evaluating
    "n_ft_weights": 157,  # number of weights ft steps, 157 is one epoch

    # ws training stuff
    "n_monte": 1,  # number of architecture sampled in each batch
    "prorata": False,  # sample prorata to number of parameters
    "single_k": False,  # use single-path
    "cutout": False,  # use cutout

    # supernet stuff
    "c_init": 16,  # number of channels of the first convolution
    "input_channels": 3,
    "batch_size": 256,

    # conv layers parameters
    "layers_params": {
        "conv_bias": False,  # bias on conv layers
        "affine_bn": True,  # affine parameters on bn layers
        "bn_momentum": 0.997,  # momentum on bn layers
        "bn_eps": 1e-5,  # eps on bn layers
    },

    # topology of the super-net
    "n_cells": 3,  # number of cells per stack
    "n_stacks": 3,  # number of stacks
    "n_classes": 10,
    "n_nodes_max": 7,

    # optimization stuff
    "optimizer": TFRMSprop,  # modified RMSprop to be similar to the one used in tensorflow
    "optimizer_params": {
        "lr": 0.2,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "eps": 1.0,
    },
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "criterion": torch.nn.CrossEntropyLoss(),
}


def get_n_out_min(ss):
    """
    Returns the value of the minimum numbers of edges going to the output
    in the corresponding search space, to avoid creating parameters for nothing
    :param ss: 
    :return: 
    """
    n = ss.split("_")[0]
    if n == "full":
        n = 1
    else:
        n = int(n)
    return n


def get_data_iters(params):
    """
    Creates the train and validation data iterators by splitting the cifar-10 training set
    :param params:
    :return:
    """

    # Loading Data
    train_transform, valid_transform = data_transforms_cifar10(cutout=params["cutout"], cutout_length=16)
    dataset_train = datasets.CIFAR10("./datasets", train=True, download=True, transform=train_transform)
    dataset_valid = datasets.CIFAR10("./datasets", train=True, download=True, transform=valid_transform)

    # training set contains 40,000 images, validation and test set contain 10,000 images
    dataset_valid = Subset(dataset_valid, range(4 * len(dataset_train) // 5, len(dataset_train)))
    dataset_train = Subset(dataset_train, range(4 * len(dataset_train) // 5))

    # building cyclic iterators over the training and validation sets
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=params["batch_size"], shuffle=True)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=params["batch_size"], shuffle=True)

    print("Length of datasets: Train: {}, Valid: {}".format(len(dataset_train), len(dataset_valid)))
    print("Length of loaders: Train: {}, Valid: {}".format(len(loader_train), len(loader_valid)))

    return loader_train, loader_valid


if __name__ == "__main__":

    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument("--search_space", type=str, required=True)
    parser.add_argument("--snapshot_path", type=str, default=None, required=False)
    parser.add_argument("--train_only", dest="train_only", action="store_true")
    parser.add_argument("--eval_only", dest="eval_only", action="store_true")
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    # getting the value of n_min
    search_space = args.search_space
    n_out_min = get_n_out_min(search_space)
    exp_params["n_out_min"] = n_out_min

    # creating the super-net
    model = GraphComp(**exp_params)
    print("Number of trainable parameters: {}".format(
        sum([p.numel() for p in model.parameters()])))
    if args.snapshot_path is not None:
        model.load_state_dict(torch.load(args.snapshot_path, map_location=exp_params["device"]))

    # data iterators
    loader_train, loader_valid = get_data_iters(exp_params)

    # load the architecture dataset
    dataset_path = "./datasets/sp_{}.pkl".format(search_space)

    # folder where results are saved
    model_dir = get_output_folder(os.path.join("./results", exp_params["output_dir"]), search_space)

    # If training, skipped if only evaluating
    if not args.eval_only:
        # create the sampler
        task_sampler = GraphSampler(**{
            "dataset_path": dataset_path,
            "n_nodes": exp_params["n_nodes_max"],
            "prorata": exp_params["prorata"],
        })
        train_supernet(model_dir, model, task_sampler=task_sampler,
                       train_iter=loader_train, valid_iter=loader_valid,
                       exp_params=exp_params)

    # If evaluating, skipped if only training
    if not args.train_only:
        # create the sampler
        task_sampler = GraphSampler(**{
            "dataset_path": dataset_path,
            "n_nodes": exp_params["n_nodes_max"],
            "prorata": False,
        })
        eval_supernet(model_dir, model, task_sampler=task_sampler,
                      train_iter=loader_train, valid_iter=loader_valid,
                      exp_params=exp_params)
