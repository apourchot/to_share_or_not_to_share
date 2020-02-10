import os
import pickle
import time
from copy import deepcopy

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import set_seed


def train_supernet(model_dir, model, task_sampler, train_iter, valid_iter, exp_params):
    """
    :param model_dir:
    :param model:
    :param task_sampler:
    :param train_iter:
    :param valid_iter:
    :param exp_params:
    :return:
    """
    writer = None
    since = time.time()

    seed = set_seed(exp_params["train_seed"])
    exp_params["train_seed"] = seed
    with open(os.path.join(model_dir, "params_dict.pkl"), 'wb') as f:
        pickle.dump(exp_params, f)

    # optimizer and scheduler
    optimizer = exp_params["optimizer"](model.parameters(), **exp_params["optimizer_params"])
    scheduler = exp_params["scheduler"](optimizer, T_max=exp_params["num_epochs"] * len(train_iter))

    # metrics
    total_metrics = {
        "train": [],
        "valid": [],
    }

    # data iterators
    iters = {"train": train_iter, "valid": valid_iter}

    # training
    for epoch in range(exp_params["num_epochs"]):

        print("-" * 100)
        print("Iter Epoch {}/{}".format(epoch + 1, exp_params["num_epochs"]))
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
                    tasks = task_sampler.sample(n_monte=exp_params["n_monte"])
                    loss_t = None
                    accs_t = []

                    for task in tasks:

                        # forward
                        x_t, y_t = x.to(exp_params["device"]), y.to(exp_params["device"])
                        preds_t = model.forward(x_t, task)

                        # computing gradient
                        if loss_t is None:
                            loss_t = exp_params["criterion"](preds_t, y_t) / exp_params["n_monte"]
                        else:
                            loss_t += exp_params["criterion"](preds_t, y_t) / exp_params["n_monte"]

                        # saving accuracies
                        accs_t.append(np.mean((torch.max(preds_t, dim=1)[1] == y_t).cpu().numpy()))

                    # update
                    loss_t.backward()
                    optimizer.step()
                    scheduler.step()
                    model.none_grad()

                    # adding metrics
                    epoch_metrics[phase]["learning_rate"].append(scheduler.get_lr())
                    epoch_metrics[phase]["losses_train"].append(loss_t.item())
                    epoch_metrics[phase]["accs_train"].append(np.mean(accs_t))

                elif exp_params["perform_valid"]:

                    model.eval()
                    task = task_sampler.sample(n_monte=exp_params["n_monte"])[0]

                    # forward
                    x_v, y_v = x.to(exp_params["device"]), y.to(exp_params["device"])
                    with torch.no_grad():
                        preds_v = model.forward(x_v, task)
                        loss_v = exp_params["criterion"](preds_v, y_v)

                    # adding metrics
                    epoch_metrics[phase]["losses_valid"].append(loss_v.item())
                    epoch_metrics[phase]["accs_valid"].append(
                        np.mean((torch.max(preds_v, dim=1)[1] == y_v).cpu().numpy()))

                else:
                    break

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
        if exp_params["use_tensorboard"]:
            if exp_params["use_tensorboard"] and writer is None:
                writer = SummaryWriter(model_dir)
            for phase in ["train", "valid"]:
                for key, value in epoch_metrics[phase].items():
                    if value is not None:
                        writer.add_scalar(phase + "/" + key, value, epoch)

        time_elapsed = time.time() - since
        print(to_print + "Time Elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        # save everything
        if exp_params["save"] and ((epoch + 1) % exp_params["save_period"] == 0):

            # saving model
            weights_path = os.path.join(model_dir, "model_weights_epoch_{0}_of_{1}.pth".
                                        format(epoch + 1, exp_params["num_epochs"]))
            torch.save(model.state_dict(), weights_path)

            # saving stuff to retrieve
            with open(os.path.join(model_dir, "total_metrics.pkl"), "wb") as handle:
                pickle.dump(total_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    return total_metrics


def ft_bn_stats(model, task, train_iter, exp_params):
    """
    Fine-tune the batch norm stats to the task at hand
    :param model:
    :param task:
    :param train_iter:
    :param exp_params:
    :return:
    """
    model.train()  # in order to update bn stats
    model.set_bn_momentum(1.0)  # so no data leaks from a batch to the other

    # given task
    matrix, ops_list = task

    # compute bn stats with given task
    bn_means = {n: torch.zeros_like(m.running_mean) for (n, m) in model.named_modules() if "BatchNorm" in str(type(m))}
    bn_vars = {n: torch.zeros_like(m.running_var) for (n, m) in model.named_modules() if "BatchNorm" in str(type(m))}

    # we don't need gradients
    with torch.no_grad():
        for n, (x_t, y_t) in enumerate(train_iter):

            if n < exp_params["n_ft_bn_stats"]:
                x_t, y_t = x_t.to(exp_params["device"]), y_t.to(exp_params["device"])
                model.forward(x_t, (matrix, ops_list))

                for name, module in model.named_modules():
                    if "BatchNorm" in str(type(module)):
                        bn_means[name] += module.running_mean / exp_params["n_ft_bn_stats"]
                        bn_vars[name] += module.running_var / exp_params["n_ft_bn_stats"]

            else:
                break

    model.set_bn_momentum(exp_params["layers_params"]["bn_momentum"])  # reset momentum to initial value


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

    # given task
    matrix, ops_list = task

    # temporary optimizer
    optimizer = exp_params["optimizer"](model.parameters(), **exp_params["optimizer_params"])
    scheduler = exp_params["scheduler"](optimizer, T_max=exp_params["n_ft_weights"])

    n_steps = 0
    while n_steps < exp_params["n_ft_weights"]:
        for x_t, y_t in train_iter:

            x_t, y_t = x_t.to(exp_params["device"]), y_t.to(exp_params["device"])
            preds_t = model.forward(x_t, (matrix, ops_list))
            loss_t = exp_params["criterion"](preds_t, y_t)

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            scheduler.step()

            n_steps += 1
            if n_steps >= exp_params["n_ft_weights"]:
                break


def eval_model(model, task, valid_iter, exp_params):
    """
    Evaluate a super-net on a single model
    :param model:
    :param task:
    :param valid_iter:
    :param exp_params:
    :return:
    """
    model.eval()
    matrix, ops_list = task

    total_correct_v = []
    with torch.no_grad():
        for x_v, y_v in valid_iter:
            x_v, y_v = x_v.to(exp_params["device"]), y_v.to(exp_params["device"])
            preds_v = model.forward(x_v, task=(matrix, ops_list))
            total_correct_v.extend((torch.max(preds_v, dim=1)[1] == y_v).cpu().numpy())
    accs = np.mean(total_correct_v)

    return accs


def eval_supernet(model_dir, model, task_sampler, train_iter, valid_iter, exp_params):
    """
    Evaluates the supernet on several sampled models
    :param model_dir:
    :param model:
    :param task_sampler:
    :param train_iter:
    :param valid_iter:
    :param exp_params:
    :return:
    """
    seed = set_seed(exp_params["eval_seed"])  # to retrieve the sampled models
    exp_params["eval_seed"] = seed
    with open(os.path.join(model_dir, "params_dict_eval.pkl"), 'wb') as f:
        pickle.dump(exp_params, f)

    # current state dict of the super net, it is saved here and loaded before
    # each evaluation to ensure independent evaluations of the models
    state_dict = deepcopy(model.state_dict())

    # evaluating n_rand_eval models
    results = []
    for _ in tqdm(range(exp_params["n_rand_evals"]), ncols=100):

        # reload load params of the supernet, including bn stats
        model.load_state_dict(state_dict)

        # sample a model to evaluate
        matrix, ops_list, metrics = task_sampler.sample(n_monte=1, return_metrics=True)[0]

        # perform the necessary fine-tuning
        if exp_params["ft_bn_stats"]:
            ft_bn_stats(model, (matrix, ops_list), train_iter, exp_params)
        if exp_params["ft_weights"]:
            ft_weights(model, (matrix, ops_list), train_iter, exp_params)

        # perform eval
        valid_accs = eval_model(model, (matrix, ops_list), valid_iter, exp_params)

        # appending resulting metrics
        bench_metrics = metrics[1][108]
        proxy_metrics = {"final_validation_accuracy": valid_accs}
        res = {
            "architecture": (matrix, ops_list),
            "full_metrics": metrics,
            "bench_metrics": bench_metrics,
            "proxy_metrics": proxy_metrics,
        }
        results.append(res)

    # save the results
    with open(os.path.join(model_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(results, f)
