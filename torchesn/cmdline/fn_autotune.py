'''

This is a commandline utility tool, that helps you to use evolutionary genetic search, to find good hyperparameters for your ESN models.
It will automate calling out to DEAP (Distributed Evolutionary Algorithms in Python) and configuring it with good defaults.

Only the most useful parameters are exposed here, to simplify the modelling searching process.

Notes:
You can set the worker pool information here, controlling how many concurrent processes will be used by Deap. 
This needs setting by you after some experimentation - to discover how much GPU memory your training processes consume. If your consumed_memory * workers > GPU_memory your job will fail.

to do:

Later I will parameterise the cmdline tool used to run the evolution, so we many different types of models, that share a common data prep pipeline.

'''
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
@click.option("--min_layers", default=1, type=int, help="the minimum number of layers to use in our DeepESN search. default: 1")
@click.option("--hidden_size_low", default=500, help="lower bound of the hidden reservoir to search: default 500")
@click.option("--hidden_size_high", default=1000, help="upper bound of the hidden size to search. Default 2000")

@click.option("--population", default=30, help="The size of the genetic population to evolve, bigger will find better: default 30")
@click.option("--number_islands", default=0, help="BETA: The number of islands used in evolution (each of population_size): default 0")
@click.option("--generations", default=20, help="the number of generations to evolve the population, longer may find better: default 20")
@click.option("--worker_pool", default=1, type=int, help="the number of concurrent workers in the pool for the evolution. Ensure you do not max your GPU memory out!: default 1")
@click.option("--header", type = click.BOOL, default=False, help="a switch to print the header record to interpret the results. Default: False")
@click.option("--func", default="fn_cotrend", help="this indicates which commandline fn to call to execute the ESN runs. Default: fn_cotrend")

def tuneESN(dataset, input_size, output_size, batch_first, max_layers, min_layers, hidden_size_low, hidden_size_high, population, generations, header, worker_pool, number_islands, func):
     #     Above we created just enough commandline parameters to run the autotuning. 
     #     I have set smart search defaults for all the ESN search hyperparameters.
     #     Due to cost implications, I'm leaving in hidden_low, hidden_high, population size, and number of generations for you to override.
     #     now, call the DEAP genetic search, with our parameters
     best_params = defineSearch(dataset, input_size, output_size, batch_first
                               , population_size=population, number_of_generations=generations
                               , search_max_num_layers=max_layers, search_min_num_layers=min_layers
                               , search_hidden_size_low=hidden_size_low, search_hidden_size_high=hidden_size_low
                               , pool_size=worker_pool, number_islands=number_islands, cmdline_tool=func)


     # pretty print the final data frrom the autotuning run
     pp = pprint.PrettyPrinter(indent=4)
     pp.pprint(best_params)

     handy_cmd = ("# " + best_params['_cmdline_tool'] + " --hidden_size " + str(best_params['attr_hidden']) + " --input_size " + str(best_params['attr_input_size'])
                  + " --output_size " + str(best_params['attr_output_size']) + " --spectral_radius " + str(best_params['attr_spectral_radius']) 
                  + " --density " + str(best_params['attr_density']) + " --leaking_rate " + str(best_params['attr_leaking_rate'])
                  + " --lambda_reg " + str(best_params['attr_lambda_reg']) + " --nonlinearity " + str(best_params['attr_nonlinearity'])
                  + " --readout_training " + str(best_params['attr_readout_training']) + " --w_io " + str(best_params['attr_w_io'])
                  + " --w_ih_scale " + str(best_params['attr_w_ih_scale']) + " --seed 10" )
     print("")
     print("# the command line view of the params is:")
     print(handy_cmd)
     return
    

if __name__ == "__main__":
     tuneESN()

