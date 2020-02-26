from models.nasbench_101.graph_comps import Graph as GraphModel101
from models.nasbench_101.graph_sampler import GraphSampler as GraphSampler101
from models.nasbench_201.graph_comps import Graph as GraphModel201
from models.nasbench_201.graph_sampler import GraphSampler as GraphSampler201


def graph_model_101_factory(config):
    model = GraphModel101(**config)
    return model


def graph_model_201_factory(config):
    model = GraphModel201(**config)
    return model


def model_factory(config):

    meta = {
        "GraphModel101": graph_model_101_factory,
        "GraphModel201": graph_model_201_factory,
    }

    return meta[config["name"]](config)


def graph_sampler_101_factory(config):
    model = GraphSampler101(**config)
    return model


def graph_sampler_201_factory(config):
    model = GraphSampler201(**config)
    return model


def graph_sampler_factory(config):

    meta = {
        "GraphSampler101": graph_sampler_101_factory,
        "GraphSampler201": graph_sampler_201_factory,
    }

    return meta[config["name"]](config)
