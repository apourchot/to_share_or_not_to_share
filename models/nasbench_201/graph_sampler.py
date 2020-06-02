import pickle

import numpy as np


class GraphSampler(object):
    """
    Sample architectures uniformly from available dataset
    """
    def __init__(self, dataset_path, **kwargs):
        super(GraphSampler, self).__init__()

        # load dataset
        with open(dataset_path, "rb") as f:
            self.dataset = pickle.load(f)
        print("Number of architectures: {}".format(len(self.dataset)))

        # build probs
        self.probs = np.ones(len(self.dataset))
        self.probs = self.probs / np.sum(self.probs)
        print("Probs: ", self.probs)

    def sample(self, n_monte=1, return_metrics=False):
        """
        Samples n_monte architecture
        :param n_monte:
        :param return_metrics: whether to return the metrics associated with the architecture
        :return: a list of matrices and operations describing
        the sampled architecture
        """
        matrices = []
        metrics = []

        # sampling an architecture n_monte times
        for n in range(n_monte):

            # sampling datum
            data = self.dataset[np.random.choice(len(self.dataset), p=self.probs)]

            # matrix used for all tasks
            matrix = data["architecture"]
            matrices.append(matrix)

            # append metrics if necessary
            if return_metrics:
                metrics.append(data)

        if return_metrics:
            return list(zip(matrices, metrics))
        return list(matrices)

    def get(self, n, return_metrics):
        """
        Returns the nth architecture
        :param n:
        :param return_metrics: whether to return the metrics associated with the architecture
        :return: a list of matrices and operations describing
        the sampled architecture
        """
        assert n < len(self.dataset)
        matrices = []
        metrics = []

        # sampling datum
        data = self.dataset[n]

        # matrix used for all tasks
        matrix = data["architecture"]
        matrices.append(matrix)

        # append metrics if necessary
        if return_metrics:
            metrics.append(data)

        if return_metrics:
            return list(zip(matrices, metrics))
        return list(matrices)

    def get_all(self, return_metrics):

        matrices = []
        metrics = None

        if return_metrics:
            metrics = []

        # sampling an architecture n_monte times
        for n in range(len(self.dataset)):

            # sampling datum
            data = self.dataset[n]

            # matrix used for all tasks
            matrix = data["architecture"]
            matrices.append(matrix)

            # append metrics if necessary
            if return_metrics:
                metrics.append(data)

        if return_metrics:
            return list(zip(matrices, metrics))
        return list(matrices)

    def get_random_perm(self):
        return np.random.permutation(range(len(self.dataset)))
