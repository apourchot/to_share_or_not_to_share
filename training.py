import os
import yaml
import time
import pickle

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from copy import deepcopy
from utils import get_scheduler, get_optimizer, get_criterion
from utils import set_seed

from itertools import cycle


def time_profiling(results_dir, model, task_sampler, loader_train, device, config):
    """
    Measure the time necessary to perform forward / backward prop on current device
    :param results_dir:
    :param model:
    :param task_sampler:
    :param loader_train:
    :param device:
    :param config:
    :return:
    """
    # creating input
    all_tasks = task_sampler.get_all(return_metrics=False)
    cycle_train = cycle(loader_train)

    # training stuff
    optimizer = get_optimizer(model.parameters(), config["TRAIN"]["OPTIMIZER"])
    criterion = get_criterion(config["TRAIN"]["CRITERION"])

    # output
    times_forward = []
    times_backward = []

    # warm-up
    for i in tqdm(range(10), ncols=100):
        task = all_tasks[i]
        x, y = next(cycle_train)
        x, y = x.to(device), y.to(device)

        preds = model(x, task)
        loss = criterion(preds, y)
        loss.backward()

    for i in tqdm(range(len(all_tasks)), ncols=100):
        task = all_tasks[i]
        x, y = next(cycle_train)
        x, y = x.to(device), y.to(device)

        # forward time
        t_f = time.time()
        pred = model(x, task)
        dt_f = time.time() - t_f
        times_forward.append(dt_f)

        # backward time
        t_b = time.time()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        dt_b = time.time() - t_b
        times_backward.append(dt_b)

    with open(os.path.join(results_dir, "forward_times.pkl"), "wb") as f:
        pickle.dump(times_forward, f)
    with open(os.path.join(results_dir, "backward_times.pkl"), "wb") as f:
        pickle.dump(times_backward, f)


def train_supernet(results_dir, model, task_sampler, train_iter, valid_iter, device, config):
    """
    :param results_dir:
    :param model:
    :param task_sampler:
    :param train_iter:
    :param valid_iter:
    :param device:
    :param config:
    :return:
    """
    writer = None
    since = time.time()

    seed = set_seed(config["TRAIN"]["train_seed"])
    config["TRAIN"]["train_seed"] = seed
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # metrics
    total_metrics = {
        "train": [],
        "valid": [],
    }

    # data iterators
    iters = {"train": train_iter, "valid": valid_iter}

    # training stuff
    optimizer = get_optimizer(model.parameters(), config["TRAIN"]["OPTIMIZER"])
    scheduler = get_scheduler(optimizer, config["TRAIN"]["SCHEDULER"])
    criterion = get_criterion(config["TRAIN"]["CRITERION"])

    # training
    for epoch in range(config["TRAIN"]["num_epochs"]):

        print("-" * 100)
        print("Iter Epoch {}/{}".format(epoch + 1, config["TRAIN"]["num_epochs"]))
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
                    tasks = task_sampler.sample(n_monte=config["TRAIN"]["GRAPH_SAMPLER"]["n_monte"])
                    loss_t = None
                    accs_t = []

                    for task in tasks:

                        # forward
                        x_t, y_t = x.to(device), y.to(device)
                        preds_t = model.forward(x_t, task)

                        # computing gradient
                        if loss_t is None:
                            loss_t = criterion(preds_t, y_t) / config["TRAIN"]["GRAPH_SAMPLER"]["n_monte"]
                        else:
                            loss_t += criterion(preds_t, y_t) / config["TRAIN"]["GRAPH_SAMPLER"]["n_monte"]

                        # saving accuracies
                        accs_t.append(np.mean((torch.max(preds_t, dim=1)[1] == y_t).cpu().numpy()))

                    # update
                    loss_t.backward()
                    optimizer.step()
                    scheduler.step(epoch)
                    model.none_grad()

                    # adding metrics
                    epoch_metrics[phase]["learning_rate"].append(scheduler.get_lr())
                    epoch_metrics[phase]["losses_train"].append(loss_t.item())
                    epoch_metrics[phase]["accs_train"].append(np.mean(accs_t))

                elif config["TRAIN"]["perform_valid"]:

                    model.eval()
                    task = task_sampler.sample()[0]

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
        if config["TRAIN"]["use_tensorboard"]:
            if config["TRAIN"]["use_tensorboard"] and writer is None:
                writer = SummaryWriter(results_dir)
            for phase in ["train", "valid"]:
                for key, value in epoch_metrics[phase].items():
                    if value is not None:
                        writer.add_scalar(phase + "/" + key, value, epoch)

        time_elapsed = time.time() - since
        print(to_print + "Time Elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        # save everything
        if config["TRAIN"]["save"] and ((epoch + 1) % config["TRAIN"]["save_period"] == 0):

            # saving model
            weights_path = os.path.join(results_dir, "model_weights_epoch_{0}_of_{1}.pth".
                                        format(epoch + 1, config["TRAIN"]["num_epochs"]))
            torch.save(model.state_dict(), weights_path)

            # saving stuff to retrieve
            with open(os.path.join(results_dir, "total_metrics.pkl"), "wb") as handle:
                pickle.dump(total_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    return total_metrics


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


def eval_supernet(results_dir, model, task_sampler, train_iter, valid_iter, device, config):
    """
    Evaluates the supernet on several sampled models
    :param results_dir:
    :param model:
    :param task_sampler:
    :param train_iter:
    :param valid_iter:
    :param device:
    :param config:
    :return:
    """
    seed = set_seed(config["EVAL"]["eval_seed"])  # to retrieve the sampled models
    config["EVAL"]["eval_seed"] = seed
    with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    # current state dict of the super net, it is saved here and loaded before
    # each evaluation to ensure independent evaluations of the models
    state_dict = deepcopy(model.state_dict())

    # sample models without replacement
    random_perm = task_sampler.get_random_perm()

    # evaluating n_rand_eval models
    results = []
    for i in tqdm(range(max(config["EVAL"]["n_rand_evals"]), len(random_perm)), ncols=100):

        # reload load params of the supernet, including bn stats
        model.load_state_dict(state_dict)

        # sample a model to evaluate
        task, metrics = task_sampler.get(n=random_perm[i], return_metrics=True)[0]

        # perform the necessary fine-tuning
        ft_dt = time.time()
        if config["EVAL"]["ft_bn_stats"]:
            ft_bn_stats(model, task, train_iter, device, config)
        if config["EVAL"]["ft_weights"]:
            ft_weights(model, task, train_iter, device, config)
        ft_dt = time.time() - ft_dt

        # perform eval
        eval_dt = time.time()
        valid_accs = eval_model(model, task, valid_iter, device)
        eval_dt = time.time() - eval_dt

        # appending resulting metrics
        proxy_metrics = {"final_validation_accuracy": valid_accs}
        res = {
            "architecture": task,
            "bench_metrics": metrics,
            "proxy_metrics": proxy_metrics,
            "ft_dt": ft_dt,
            "eval_dt": eval_dt,
        }
        results.append(res)

    # save the results
    with open(os.path.join(results_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(results, f)


def ft_bn_stats(model, task, train_iter, device, config):
    """
    Fine-tune only the batch norm stats to the task at hand
    :param model:
    :param task:
    :param train_iter:
    :param device:
    :param config:
    :return:
    """
    model.train()  # in order to update bn stats
    model.set_bn_momentum(1.0)  # so no data leaks from one batch to the other

    # compute bn stats with given task
    bn_means = {n: torch.zeros_like(m.running_mean) for (n, m) in model.named_modules() if "BatchNorm" in str(type(m))}
    bn_vars = {n: torch.zeros_like(m.running_var) for (n, m) in model.named_modules() if "BatchNorm" in str(type(m))}

    # we don't need gradients
    with torch.no_grad():
        for n, (x_t, y_t) in enumerate(train_iter):

            if n < config["EVAL"]["n_ft_bn_stats"]:
                x_t, y_t = x_t.to(device), y_t.to(device)
                model.forward(x_t, task)

                for name, module in model.named_modules():
                    if "BatchNorm" in str(type(module)):
                        bn_means[name] += module.running_mean / config["EVAL"]["n_ft_bn_stats"]
                        bn_vars[name] += module.running_var / config["EVAL"]["n_ft_bn_stats"]

            else:
                break

    # reset momentum to initial value
    model.set_bn_momentum(config["TRAIN"]["GRAPH_MODEL"]["layers_params"]["bn_momentum"])


def ft_weights(model, task, train_iter, device, config):
    """
    Fine-tune the weights of the super-net to the task at hand
    :param model:
    :param task:
    :param train_iter:
    :param device:
    :param config:
    :return:
    """
    # fine-tuning stuff
    model.train()

    # optimizer
    optimizer = get_optimizer(model.parameters(), config["TRAIN"]["OPTIMIZER"])

    # scheduler
    scheduler_config = {
        "name": config["TRAIN"]["SCHEDULER"]["name"],
        "T_max": config["EVAL"]["n_ft_weights"]
    }
    scheduler = get_scheduler(optimizer, scheduler_config)

    # criterion
    criterion = get_criterion(config["TRAIN"]["CRITERION"])

    n_steps = 0
    while n_steps < config["EVAL"]["n_ft_weights"]:
        for x_t, y_t in train_iter:

            x_t, y_t = x_t.to(device), y_t.to(device)
            preds_t = model.forward(x_t, task)
            loss_t = criterion(preds_t, y_t)

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            scheduler.step()

            n_steps += 1
            if n_steps >= config["EVAL"]["n_ft_weights"]:
                break
