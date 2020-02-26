import pickle

import numpy as np


class GraphSampler(object):
    """
    Sample architectures uniformly from available dataset
    """
    def __init__(self, dataset_path, n_nodes, prorata=False, **kwargs):
        super(GraphSampler, self).__init__()

        # Topology of the cell
        self.n_nodes = n_nodes

        # load dataset
        with open(dataset_path, "rb") as f:
            self.dataset = pickle.load(f)
        print("Number of architectures: {}".format(len(self.dataset)))

        # build probs
        self.prorata = prorata
        if prorata:  # sample prorata to number of params
            self.probs = np.array([d[0]["trainable_parameters"] for d in self.dataset])
        else:
            self.probs = np.ones(len(self.dataset))
        self.probs = self.probs / np.sum(self.probs)
        print("Probs: ", self.probs)

        # idx of each ops
        self.ops_idx = {
            "maxpool3x3": 0,
            "conv1x1-bn-relu": 1,
            "conv3x3-bn-relu": 2,
        }

    def sample(self, n_monte=1, return_metrics=False):
        """
        Samples n_monte architecture
        :param n_monte:
        :param return_metrics: whether to return the metrics associated with the architecture
        :return: a list of matrices and operations describing
        the sampled architecture
        """
        matrices = []
        ops_list = []
        metrics = None

        if return_metrics:
            metrics = []

        # sampling an architecture n_monte times
        for n in range(n_monte):

            # sampling datum
            data = self.dataset[np.random.choice(len(self.dataset), p=self.probs)]

            # matrix used for all tasks
            matrix = data[0]["module_adjacency"]
            matrices.append(matrix)

            # operation to be used
            module_ops = data[0]["module_operations"]
            n_ops = len(module_ops)
            ops = np.zeros(n_ops - 2)
            for i in range(n_ops - 2):
                ops[i] = self.ops_idx[module_ops[i + 1]]
            ops_list.append(ops)

            # append metrics if necessary
            if return_metrics:
                metrics.append(data)

        if return_metrics:
            return list(zip(zip(matrices, ops_list), metrics))
        return list(zip(matrices, ops_list))

    def train(self):
        if self.prorata:  # sample prorata to number of params
            self.probs = np.array([d[0]["trainable_parameters"] for d in self.dataset])
        self.probs = self.probs / np.sum(self.probs)

    def eval(self):
        self.probs = np.ones(len(self.dataset))
        self.probs = self.probs / np.sum(self.probs)
