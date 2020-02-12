# To Share or Not To Share: A Comprehensive Appraisal of Weight-Sharing

This directory contains all the code necessary to reproduce the results of the paper: https://arxiv.org/abs/2002.04289v1

__Main Dependencies:__
* PyTorch (https://github.com/pytorch/pytorch)
* NASBench (https://github.com/google-research/nasbench)
* TensorboardX (https://github.com/lanpa/tensorboardX)
* tqdm (https://github.com/tqdm/tqdm)
* seaborn (https://github.com/mwaskom/seaborn)
* graphviz (https://github.com/xflr6/graphviz)

## Steps to reproduce the results:
1. Download the nasbench dataset and put under the "datasets" folder
2. Run all the cells of the notebook "datasets.ipynb": this will create all the different search spaces
3. Run individual trainings and evaluations with `python main.py --search_space SEARCH_SPACE` after having set 
parameters as wanted in "main.py". Seeds used to generate the results of the paper can be found in the "figures.ipynb" notebook.
4. generate the different figures using the "figures.ipnyb" notebook, after having arranged the files 
resulting from the evaluations as in the "results_paper" directory

## Visualization
* Examples of architectures coming from each dataset can be generated at the end of the "datasets.ipynb" notebook.

* All the curves of the paper (and more) can be generated and observed in the "figures.ipnyb" notebook.
