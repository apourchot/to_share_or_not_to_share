import os
import pickle
import time
from copy import deepcopy

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import set_seed


def train_supernet(model_dir, model, graph_s, criterion, optimizer, scheduler, train_iter, valid_iter, config, device):
    """
    :param model_dir:
    :param model:
    :param graph_s:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param train_iter:
    :param valid_iter:
    :param config:
    :param device:
    :return:
    """
    writer = None
    since = time.time()

    seed = set_seed(config["train_seed"])
    config["train_seed"] = seed
    with open(os.path.join(model_dir, "train_config.yaml"), "w") as f:
        yaml.dump(config, f)

    # metrics
    total_metrics = {
        "train": [],
        "valid": [],
    }

    # data iterators
    iters = {"train": train_iter, "valid": valid_iter}

    # training
    for epoch in range(config["num_epochs"]):

        print("-" * 100)
        print("Iter Epoch {}/{}".format(epoch + 1, config["num_epochs"]))
        print("-" * 100)

        epoch_metrics = {
            "train": {
                "learning_rate": [],
                "losses_train": [],
                "accs_train": [],
            },
            "valid": {
                "losses_valid": [],
                "accs_valid": [],
            }
        }

        for phase in ["train", "valid"]:

            for iter_cpt, (x, y) in tqdm(enumerate(iters[phase]), ncols=100, total=len(iters[phase])):

                # perform an update
                if phase == "train":

                    model.train()
                    graph_s.train()

                    tasks = graph_s.sample(n_monte=config["GRAPH_SAMPLER"]["n_monte"])
                    loss_t = None
                    accs_t = []

                    for task in tasks:

                        # forward
                        x_t, y_t = x.to(device), y.to(device)
                        preds_t = model.forward(x_t, task)

                        # computing gradient
                        if loss_t is None:
                            loss_t = criterion(preds_t, y_t) / config["GRAPH_SAMPLER"]["n_monte"]
                        else:
                            loss_t += criterion(preds_t, y_t) / config["GRAPH_SAMPLER"]["n_monte"]

                        # saving accuracies
                        accs_t.append(np.mean((torch.max(preds_t, dim=1)[1] == y_t).cpu().numpy()))

                    # update
                    loss_t.backward()
                    optimizer.step()
                    model.none_grad()

                    # adding metrics
                    epoch_metrics[phase]["learning_rate"].append(scheduler.get_lr())
                    epoch_metrics[phase]["losses_train"].append(loss_t.item())
                    epoch_metrics[phase]["accs_train"].append(np.mean(accs_t))

                elif config["perform_valid"]:

                    model.eval()
                    graph_s.eval()

                    task = graph_s.sample(n_monte=1)

                    # forward
                    x_v, y_v = x.to(device), y.to(device)
                    with torch.no_grad():
                        preds_v = model.forward(x_v, task)
                        loss_v = criterion(preds_v, y_v)

                    # adding metrics
                    epoch_metrics[phase]["losses_valid"].append(loss_v.item())
                    epoch_metrics[phase]["accs_valid"].append(
                        np.mean((torch.max(preds_v, dim=1)[1] == y_v).cpu().numpy()))

                else:
                    break

        scheduler.step()

        # average metrics over epoch
        to_print = "\n"
        for phase in ["train", "valid"]:
            to_print += phase.upper() + ":\n"
            for key in epoch_metrics[phase].keys():
                if len(epoch_metrics[phase][key]) > 0:
                    epoch_metrics[phase][key] = np.mean(epoch_metrics[phase][key])
                    to_print += "{}: {:.4f}".format(key, epoch_metrics[phase][key]) + "\n"
                else:
                    epoch_metrics[phase][key] = None
            total_metrics[phase].append(epoch_metrics[phase])
            to_print += "\n"

        # tensorboard integration to plot nice curves
        if config["use_tensorboard"]:
            if config["use_tensorboard"] and writer is None:
                writer = SummaryWriter(model_dir)
            for phase in ["train", "valid"]:
                for key, value in epoch_metrics[phase].items():
                    if value is not None:
                        writer.add_scalar(phase + "/" + key, value, epoch)

        time_elapsed = time.time() - since
        print(to_print + "Time Elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        # save everything
        if config["save"] and ((epoch + 1) % config["save_period"] == 0):

            # saving model
            weights_path = os.path.join(model_dir, "model_weights_epoch_{0}_of_{1}.pth".
                                        format(epoch + 1, config["num_epochs"]))
            torch.save(model.state_dict(), weights_path)

            # saving stuff to retrieve
            with open(os.path.join(model_dir, "total_metrics.pkl"), "wb") as handle:
                pickle.dump(total_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    return total_metrics


def ft_bn_stats(model, task, train_iter, n_ft_bn_stats, device):
    """
    Fine-tune the batch norm stats to the task at hand
    :param model:
    :param task:
    :param train_iter:
    :param n_ft_bn_stats:
    :param device:
    :return:
    """
    model.train()  # in order to update bn stats

    bn_momentum = model.get_bn_momentum()
    model.set_bn_momentum(1.0)  # so no data leaks from a batch to the other

    # compute bn stats with given task
    bn_means = {n: torch.zeros_like(m.running_mean) for (n, m) in model.named_modules() if "BatchNorm" in str(type(m))}
    bn_vars = {n: torch.zeros_like(m.running_var) for (n, m) in model.named_modules() if "BatchNorm" in str(type(m))}

    # we don't need gradients
    with torch.no_grad():
        for n, (x_t, y_t) in enumerate(train_iter):

            if n < n_ft_bn_stats:
                x_t, y_t = x_t.to(device), y_t.to(device)
                model.forward(x_t, task)

                for name, module in model.named_modules():
                    if "BatchNorm" in str(type(module)):
                        bn_means[name] += module.running_mean / n_ft_bn_stats
                        bn_vars[name] += module.running_var / n_ft_bn_stats

            else:
                break

    model.set_bn_momentum(bn_momentum)  # reset momentum to initial value


# TBD
def ft_weights(model, task, train_iter, exp_params):
    """
    Fine-tune the weights of the super-net to the task at hand
    :param model:
    :param task:
    :param train_iter:
    :param exp_params:
    :return:
    """
    # fine-tuning stuff
    model.train()

    # temporary optimizer
    optimizer = exp_params["optimizer"](model.parameters(), **exp_params["optimizer_params"])
    scheduler = exp_params["scheduler"](optimizer, T_max=exp_params["n_ft_weights"])

    n_steps = 0
    while n_steps < exp_params["n_ft_weights"]:
        for x_t, y_t in train_iter:

            x_t, y_t = x_t.to(exp_params["device"]), y_t.to(exp_params["device"])
            preds_t = model.forward(x_t, task)
            loss_t = exp_params["criterion"](preds_t, y_t)

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            scheduler.step()

            n_steps += 1
            if n_steps >= exp_params["n_ft_weights"]:
                break


def eval_model(model, task, valid_iter, device):
    """
    Evaluate a super-net on a single model
    :param model:
    :param task:
    :param valid_iter:
    :param device:
    :return:
    """
    model.eval()

    total_correct_v = []
    with torch.no_grad():
        for x_v, y_v in valid_iter:
            x_v, y_v = x_v.to(device), y_v.to(device)
            preds_v = model.forward(x_v, task)
            total_correct_v.extend((torch.max(preds_v, dim=1)[1] == y_v).cpu().numpy())
    accs = np.mean(total_correct_v)

    return accs


def eval_supernet(model_dir, model, task_sampler, train_iter, valid_iter, config, device):
    """
    Evaluates the supernet on several sampled models
    :param model_dir:
    :param model:
    :param task_sampler:
    :param train_iter:
    :param valid_iter:
    :param config:
    :param device:
    :return:
    """
    seed = set_seed(config["eval_seed"])
    config["eval_seed"] = seed
    with open(os.path.join(model_dir, "eval_config.yaml"), "w") as f:
        yaml.dump(config, f)

    # evaluating mode for task sampler i.e back to uniform sampling
    task_sampler.eval()

    # current state dict of the super net, it is saved here and loaded before
    # each evaluation to ensure independent evaluations of the models
    state_dict = deepcopy(model.state_dict())

    # evaluating n_rand_eval models
    results = []
    for _ in tqdm(range(config["n_rand_evals"]), ncols=100):

        # reload load params of the supernet, including bn stats
        model.load_state_dict(state_dict)

        # sample a model to evaluate
        task, metrics = task_sampler.sample(n_monte=1, return_metrics=True)[0]

        # perform the necessary fine-tuning
        if config["ft_bn_stats"]:
            ft_bn_stats(model, task, train_iter, config["n_ft_bn_stats"], device)
        if config["ft_weights"]:
            ft_weights(model, task, train_iter, config)

        # perform eval
        valid_accs = eval_model(model, task, valid_iter, device)

        # appending resulting metrics
        # bench_metrics = metrics[1][108]
        proxy_metrics = {"final_validation_accuracy": valid_accs}
        res = {
            "architecture": task,
            "full_metrics": metrics,
            # "bench_metrics": bench_metrics,
            "proxy_metrics": proxy_metrics,
        }
        results.append(res)

    # save the results
    with open(os.path.join(model_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(results, f)
