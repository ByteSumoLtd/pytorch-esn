
import time
import torch.nn
import numpy as np
import random
import pprint
import torchesn
from torchesn.optim import *

################# define the parameters to drive the script

import click

@click.command()

@click.option("--dataset", default="datasets/mg17.csv", help="location of the csv datafile, in the examples/data directory. Default: 'datasets/mg17.csv'")
@click.option("--input_size", default=1, help="the size of the input data, a univariate timeseries is 1: default 1")     
@click.option("--output_size", default=1, help="size input size data, a univariate timeseries is 1: default 1")     
@click.option("--batch_first", default=False, help="If True the input and output tensors are provided as (batch, seq, feature). Default: False")
@click.option("--max_layers", default=1, type=int, help="the maximum number of layers use in our DeepESN search. default: 1")
@click.option("--hidden_size_low", default=500, help="lower bound of the hidden reservoir to search: default 500")
@click.option("--hidden_size_high", default=1000, help="upper bound of the hidden size to search. Default 2000")

@click.option("--population", default=30, help="The size of the genetic population to evolve, bigger will find better: default 30")
@click.option("--generations", default=20, help="the number of generations to evolve the population, longer may find better: default 20")
@click.option("--worker_pool", default=1, type=int, help="the number of concurrent workers in the pool for the evolution. Ensure you do not max your GPU memory out!: default 1")
@click.option("--header", type = click.BOOL, default=False, help="a switch to print the header record to interpret the results. Default: False")


def tuneESN(dataset, input_size, output_size, batch_first, max_layers, hidden_size_low, hidden_size_high, population, generations, header, worker_pool):
     #     Above we created just enough commandline parameters to run the autotuning. 
     #     I have set smart search defaults for all the ESN search hyperparameters.
     #     Due to cost implications, I'm leaving in hidden_low, hidden_high, population size, and number of generations for you to override.
     #     now, call the DEAP genetic search, with our parameters
     best_params = defineSearch(dataset, input_size, output_size, batch_first
                               , population_size=population, number_of_generations=generations
                               , search_max_num_layers=max_layers 
                               , search_hidden_size_low=hidden_size_low, search_hidden_size_high=hidden_size_low
                               , pool_size=worker_pool)


     # pretty print the final data frrom the autotuning run
     pp = pprint.PrettyPrinter(indent=4)
     pp.pprint(best_params)
     return
    

if __name__ == "__main__":
     tuneESN()
