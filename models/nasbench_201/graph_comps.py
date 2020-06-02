import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nasbench_201.base_ops import (AvgPool3x3, Conv1x1, Conv3x3, Skip,
                                          Zero)
from models.nasbench_201.utils import ResNetBasicBlock, ReLUConvBN


class Edge(nn.Module):
    """
    An edge in the DAG. This represents an operation
    applied to the data.
    """
    def __init__(self, c_in, conv_bias=False, bn_affine=True, bn_momentum=0.1, bn_eps=0.00001):

        super(Edge, self).__init__()
        self.ops = nn.ModuleList([
            Zero(),
            Skip(),
            Conv1x1(c_in, c_in, conv_bias=conv_bias, bn_affine=bn_affine, bn_momentum=bn_momentum, bn_eps=bn_eps),
            Conv3x3(c_in, c_in, conv_bias=conv_bias, bn_affine=bn_affine, bn_momentum=bn_momentum, bn_eps=bn_eps),
            AvgPool3x3()
        ])

    def forward(self, x, id_op):
        """
        Forward on the node
        :param x:
        :param id_op:
        :return:
        """
        return self.ops[int(id_op)](x)


class Cell(nn.Module):
    """
    A cell is DAG comprised of a given number of node. A node is a
    representation of the data. Two nodes are linked by a Link, which
    represents the possible operations applied.
    """
    def __init__(self, n_nodes, c, layers_params):

        super(Cell, self).__init__()
        self.n_nodes = n_nodes
        self.c = c

        # all edges
        self.edges = nn.ModuleDict()
        for i in range(0, self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                self.edges[str(i) + "_" + str(j)] = Edge(self.c, layers_params)

    def forward(self, s0, matrix):
        """
        Forward pass
        :param s0:
        :param matrix: operation at each node
        :return:
        """
        # computing representations at each intermediate node
        states = [s0]
        for i in range(1, self.n_nodes):

            # summing arriving data
            data_in = []
            for j in range(0, i - 1):

                # take the intermediate representation
                h = self.edges["{}_{}".format(j, i)](states[j], id_op=matrix[i, j])

                # append to input data
                data_in.append(h)

            # sum everything and append to the list of representations
            data_in = sum(data_in) if len(data_in) > 0 else torch.zeros_like(states[0])
            states.append(data_in)

        return states[-1]


class Graph(nn.Module):
    """
    A graph describes the complete architecture: how the input is treated,
    how many cells are used, and how the output of all nodes is treated.
    """
    def __init__(self, c_input, c_init, n_classes, n_nodes, n_cells, n_stacks, layers_params, **kwargs):
        super(Graph, self).__init__()

        # super-net topology
        self.n_cells = n_cells
        self.n_stacks = n_stacks
        self.n_nodes = n_nodes

        self.c_input = c_input
        self.c_init = c_init

        # stem
        self.stem = nn.Sequential(nn.Conv2d(self.c_input, self.c_init, kernel_size=3,
                                            padding=1, bias=False),
                                  nn.BatchNorm2d(self.c_init))

        # cells
        self.cell_stacks = nn.ModuleList()
        for i in range(self.n_stacks):
            self.cell_stacks.append(nn.ModuleList([Cell(self.n_nodes, self.c_init * 2 ** i,
                                                        layers_params) for _ in range(self.n_cells)]))

        # res blocks with strides
        self.res_blocks = nn.ModuleList()
        for i in range(self.n_stacks - 1):
            self.res_blocks.append(ResNetBasicBlock(self.c_init * 2 ** i, self.c_init * 2 * 2 ** i, 2))

        # avg pool and classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.c_init * 2 ** (self.n_stacks - 1), n_classes)

    def forward(self, x, matrix):
        """
        Forward pass in the sub-graph described by the graph_path
        :param x:
        :param matrix:
        :return:
        """
        x = self.stem(x)

        for i in range(self.n_stacks - 1):
            for cell in self.cell_stacks[i]:
                x = cell(x, matrix)
            x = self.res_blocks[i](x)
        for cell in self.cell_stacks[-1]:
            x = cell(x, matrix)

        x = F.relu(x)
        x = self.global_pooling(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

    def none_grad(self, params=None):
        """
        Resets gradients to None to prevent unwanted
        updates from the optimizer with a null gradient
        :param params:
        :return:
        """
        if params is None:
            for param in self.parameters():
                param.grad = None
        else:
            for name, param in params.items():
                param.grad = None

    def get_bn_momentum(self):
        """
        Returns the current bn momentum
        """
        for name, module in self.named_modules():
            if "BatchNorm2d" in str(type(module)):
                return module.momentum

    def set_bn_momentum(self, momentum):
        """
        Sets the momentum of all bn layers to momentum
        :param momentum:
        :return:
        """
        for name, module in self.named_modules():
            if "BatchNorm2d" in str(type(module)):
                module.momentum = momentum
