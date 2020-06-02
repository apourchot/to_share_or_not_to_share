import os
from argparse import ArgumentParser

import torch
import yaml

from models.factories import graph_sampler_factory, model_factory
from training import eval_supernet, time_profiling, train_supernet
from utils import get_data_iters, get_output_folder

if __name__ == "__main__":

    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--snapshot", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--multi_gpu", dest="multi_gpu", action="store_true")

    parser.add_argument("--profile", dest="profile", action="store_true")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--eval", dest="eval", action="store_true")

    parser.add_argument("--azure", dest="azure", action="store_true")
    parser.add_argument("--launch_azure", dest="launch_azure", action="store_true")
    parser.add_argument("--multi_gpu", dest="multi_gpu", action="store_true")
    parser.add_argument("--azure_node", type=int, default=None)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    # launching the experiment on azureml
    if args.launch_azure:

        from azure_exps import launch_azure_exp
        assert args.azure_node is not None

        dargs = {
            "--config": args.config,
            "--snapshot": args.snapshot,
            "--output_dir": args.output_dir,
            "--exp_name": args.exp_name,
            "--azure": "",
        }
        if args.multi_gpu:
            dargs["--multi_gpu"] = ""
        if args.eval:
            dargs["--eval"] = ""
        if args.train:
            dargs["--train"] = ""
        if args.profile:
            dargs["--profile"] = ""

        kwargs = {
            "entry_script": "main_azure.py",
            "script_params": dargs,
            "cluster_name": "gleamer-NC12-{}".format(args.azure_node) if args.multi_gpu else "gleamer-NC6-{}".format(
                args.azure_node),
            "vm_size": "STANDARD_NC12" if args.multi_gpu else "STANDARD_NC6",
            "experiment_name": args.exp_name
        }

        run = launch_azure_exp(**kwargs)
        exit()

    else:

        # symlink if working on azure
        if args.azure:
            assert args.data_dir is not None
            os.symlink(args.data_dir, "/data")

        # load config
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print("Config:", config)

        # creating the super-net and the sampler
        model = model_factory(config["TRAIN"]["GRAPH_MODEL"])
        if args.snapshot is not None:
            model.load_state_dict(torch.load(args.snapshot_path))
        print("Number of parameters: {}".format(sum([p.numel() for p in model.parameters()])))
        task_sampler = graph_sampler_factory(config["TRAIN"]["GRAPH_SAMPLER"])

        # moving to gpu
        if args.multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            model = torch.nn.DataParallel(model)
        device = torch.cuda.current_device()
        model = model.to(device)

        # data iterators
        loader_train, loader_valid = get_data_iters(config["TRAIN"]["DATA"])

        # folder where results are saved
        model_dir = get_output_folder(args.output_dir, args.exp_name)

        # time profiling
        if args.profile:
            print("Time Profiling")
            time_profiling(model_dir, model, task_sampler,
                           loader_train, device, config)

        # If training, skipped if only evaluating
        if args.train:
            print("Training Supernet")
            train_supernet(model_dir, model, task_sampler=task_sampler,
                           train_iter=loader_train, valid_iter=loader_valid,
                           config=config, device=device)

        # If evaluating, skipped if only training
        if args.eval:
            print("Evaluating Supernet")
            eval_supernet(model_dir, model, task_sampler=task_sampler,
                          train_iter=loader_train, valid_iter=loader_valid,
                          config=config, device=device)
