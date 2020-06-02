import os

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torch.utils.data import dataset, Subset
from torchvision import datasets


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
                                    weight_decay=config["weight_decay"], nesterov=config["nesterov"])
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

        ratio = config["ratio"]

        # Loading Data
        train_transform, valid_transform = data_transforms_cifar10(cutout=False, cutout_length=16)
        dataset_train = datasets.CIFAR10("./datasets/cifar10", train=True, download=True, transform=train_transform)
        dataset_valid = datasets.CIFAR10("./datasets/cifar10", train=True, download=True, transform=valid_transform)

        # training set contains 40,000 images, validation and test set contain 10,000 images
        dataset_valid = Subset(dataset_valid, range(int(ratio * len(dataset_train)), len(dataset_train)))
        dataset_train = Subset(dataset_train, range(int(ratio * len(dataset_train))))

        # building cyclic iterators over the training and validation sets
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
        loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=config["batch_size"], shuffle=True)

        print("Length of datasets: Train: {}, Valid: {}".format(len(dataset_train), len(dataset_valid)))
        print("Length of loaders: Train: {}, Valid: {}".format(len(loader_train), len(loader_valid)))

        return loader_train, loader_valid

    else:
        raise NotImplementedError


class Cutout(object):

    def __init__(self, length):
        self.length = length

    def __call__(self, img):

        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def set_seed(seed):
    if seed is None:
        seed = np.random.randint(1e6)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def data_transforms_cifar10(cutout=False, cutout_length=None):

    cifar_mean = [0.49139968, 0.48215827, 0.44653124]
    cifar_std = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    if cutout and cutout_length is not None:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    return train_transform, valid_transform


def get_output_folder(parent_dir, run_name):
    """
    Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    run_name: str
      Name of the run

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, run_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


class TFRMSprop(Optimizer):
    """
    Implements RMSprop algorithm.
    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    The implementation here takes the square root of the gradient average AFTER
    adding epsilon (note that PyTorch interchanges these two operations).
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(TFRMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TFRMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).add_(group['eps']).sqrt_()
                else:
                    avg = square_avg.add_(group['eps']).sqrt()

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss
