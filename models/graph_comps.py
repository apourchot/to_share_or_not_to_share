import math

import torch
import torch.nn as nn

from models.graph_utils import compute_vertex_channels, get_prev_nodes
from models.ops_cnn import ConvBNRelu, MaxPool


class NodeCNN(nn.Module):
    """
    A node in the DAG. This represents an operation
    applied to the data.
    """
    def __init__(self, c_out_max, stride, single_k, layers_params):

        super(NodeCNN, self).__init__()
        if single_k:
            self.ops = nn.ModuleList([
                MaxPool(kernel_size=3, stride=stride, padding=1),
                ConvBNRelu(c_in_max=c_out_max, c_out_max=c_out_max, kernel_size=3,
                           stride=stride, padding=1, **layers_params),
            ])
        else:
            self.ops = nn.ModuleList([
                MaxPool(kernel_size=3, stride=stride, padding=1),
                ConvBNRelu(c_in_max=c_out_max, c_out_max=c_out_max, kernel_size=1,
                           stride=stride, padding=0, **layers_params),
                ConvBNRelu(c_in_max=c_out_max, c_out_max=c_out_max, kernel_size=3,
                           stride=stride, padding=1, **layers_params),
            ])
        self.single_k = single_k

    def forward(self, x, id_op, c_out):
        """
        Forward on the node
        :param x:
        :param id_op:
        :param c_out:
        :return:
        """
        if self.single_k:
            if int(id_op) == 0:
                return self.ops[int(id_op)](x, c_out)
            if int(id_op) == 1:
                return self.ops[int(id_op)](x, c_out, kernel_size=1)
            if int(id_op) == 2:
                return self.ops[1](x, c_out, kernel_size=3)
        else:
            return self.ops[int(id_op)](x, c_out)


class CellCNN(nn.Module):
    """
    A cell is DAG comprised of a given number of node. A node is a
    representation of the data. Two nodes are linked by a Link, which
    represents the possible operations applied.
    """
    def __init__(self, n_nodes_max, n_out_min, c_in, c_out, single_k, layers_params):

        super(CellCNN, self).__init__()
        self.n_nodes_max = n_nodes_max
        self.n_out_min = n_out_min
        self.c_out = c_out
        self.c_in = c_in

        # All possible projections
        self.projs = nn.ModuleDict()
        for i in range(1, n_nodes_max - 1):
            proj = ConvBNRelu(c_in_max=c_in, c_out_max=math.ceil(c_out / self.n_out_min),
                              kernel_size=1, stride=1, padding=0, **layers_params)
            self.projs[str(i)] = proj
        proj = ConvBNRelu(c_in_max=c_in, c_out_max=c_out, kernel_size=1,
                          stride=1, padding=0, **layers_params)
        self.projs[str(n_nodes_max - 1)] = proj

        # All possible nodes
        self.nodes = nn.ModuleDict()
        for i in range(1, self.n_nodes_max - 1):  # not counting input and output nodes
            node = NodeCNN(c_out_max=math.ceil(c_out / self.n_out_min), stride=1,
                           single_k=single_k, layers_params=layers_params)
            self.nodes[str(i)] = node

    def forward(self, s0, matrix, ops):
        """
        Forward pass
        :param s0:
        :param matrix: topology of the cell
        :param ops: list containing the operation to use at each node
        :return:
        """
        # Getting the number of channels at each node
        nb_cs = compute_vertex_channels(matrix=matrix, input_channels=self.c_in,
                                        output_channels=self.c_out)
        nb_nodes = matrix.shape[0]

        # computing representations at each intermediate node
        states = [s0]
        for j in range(1, nb_nodes - 1):

            # summing arriving data
            data_in = []
            prev_nodes = get_prev_nodes(matrix, j)
            for i in prev_nodes:

                # project if coming from the input
                # else take the intermediate representation
                if i == 0:
                    h = self.projs[str(j)](states[0], nb_cs[j])
                else:
                    h = states[i][:, :nb_cs[j], :, :]

                # append to input data
                data_in.append(h)
            data_in = sum(data_in) if len(data_in) > 0 else None

            # if there is no input the result is empty
            # else apply the corresponding operation
            if data_in is None:
                data_out = None
            else:
                data_out = self.nodes[str(j)](data_in, ops[j - 1], nb_cs[j])

            # append to the list of representations
            states.append(data_out)

        # concatenating nodes arriving to the last node
        # and adding to input node if necessary
        tmp = [states[i] for i in get_prev_nodes(matrix, nb_nodes - 1) if i > 0]
        if len(tmp) > 0:
            out = torch.cat(tmp, dim=1)
            if matrix[0, nb_nodes - 1]:
                out = out + self.projs[str(self.n_nodes_max - 1)](states[0], nb_cs[-1])
        else:  # rare case where only the input goes to the output (only 1 architecture in the dataset)
            out = self.projs[str(self.n_nodes_max - 1)](states[0], nb_cs[-1])
        return out


class GraphComp(nn.Module):
    """
    A graph describes the complete architecture: how the input is treated,
    how many cells are used, and how the output of all nodes is treated.
    """
    def __init__(self, c_init, input_channels, n_classes, n_cells, n_stacks, n_nodes_max,
                 n_out_min, single_k, device, layers_params, **kwargs):
        super(GraphComp, self).__init__()

        # super-net topology
        self.n_cells = n_cells
        self.n_stacks = n_stacks
        self.n_nodes_max = n_nodes_max
        self.n_out_min = n_out_min

        self.input_channels = input_channels
        self.c_in = c_init
        self.c_out = c_init

        # output
        self.n_classes = n_classes

        # stem of the network
        self.stem = ConvBNRelu(c_in_max=input_channels, c_out_max=c_init, kernel_size=3,
                               stride=1, padding=1, **layers_params)

        # cells
        self.layers = nn.ModuleList()
        for i in range(self.n_stacks):

            if i > 0:
                self.layers.append(MaxPool(kernel_size=2, stride=2, padding=0))
                self.c_out = 2 * self.c_in

            for j in range(self.n_cells):
                cell = CellCNN(self.n_nodes_max, self.n_out_min, self.c_in, self.c_out,
                               single_k=single_k, layers_params=layers_params)
                self.layers.append(cell)
                self.c_in = self.c_out

        # avg pool and classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.c_in, self.n_classes)

        self.to(device)

    def forward(self, s0, task):
        """
        Forward pass in the sub-graph described by the graph_path
        :param s0:
        :param task:
        :return:
        """
        matrix, ops = task
        s0 = self.stem(s0)
        for _, layer in enumerate(self.layers):
            s0 = layer(s0, matrix, ops)
        out = self.global_pooling(s0)
        out = self.classifier(out.view(out.size(0), -1))
        return out

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

    def set_bn_momentum(self, momentum):
        """
        Sets the momentum of all bn layers to momentum
        :param momentum:
        :return:
        """
        for name, module in self.named_modules():
            if "BatchNorm2d" in str(type(module)):
                module.momentum = momentum
