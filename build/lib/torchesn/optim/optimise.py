'''
This module is a Utility to use genetic algorithms to tune up your ESN, by finding good hyperparameters.
It is based on DEAP (Distributed Evolutionary Algorithms in Python) using a pattern delivering basic genetic search.
This helper is configured to make it easy to *integrate* the genetic tuning into your workflow, and is pre-configured to remove the boiler plate code that gets in the way. It is based on my previous prototype found here for pyESN.
https://github.com/ByteSumoLtd/pyESN/blob/master/GeneticallyTuned-pyESN-withSphericalActivations.ipynb

Still to do: do special performance tuning/caching to reduce the cost of running genetic search on multiple use cases/runs 
andrew@ybytesumo.com

'''
# Import the libraries needed to run DEAP etc
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import datetime
import math

import torch
from torchesn import nn # this is work, something else is going on... maybe I'm passing a list, not a string?
from torchesn import utils
from torchesn import seed
import time

# I'm having some issues referencing some of the functions to execute the esn ... trying out a hack here
apply_permutation = nn.reservoir.apply_permutation
AutogradReservoir = nn.reservoir.AutogradReservoir
Recurrent = nn.reservoir.Recurrent
VariableRecurrent = nn.reservoir.VariableRecurrent
StackedRNN = nn.reservoir.StackedRNN
ResTanhCell = nn.reservoir.ResTanhCell
ResReLUCell = nn.reservoir.ResReLUCell
ResIdCell = nn.reservoir.ResIdCell

class optimise():
     # Define how to call and evaluate the pytorch-esn function as an genetic
     # Individual, where hyperparamters are considered genes.
    def evaluate(individual, trX, trY, tsX, tsY, trY_flat):
        '''
        build and test a torch-esn model based on the genes of an individual, and return the MSE value
        '''
        # the fit function should get our dataset, defined as: trX, trY, tsX, tsY, trY_flat

        # extract the values for the parameters from the individual chromosome
        my_input_size = individual[0]
        my_output_size = individual[1]
        my_batch_first = individual[2]
        my_hidden_size = individual[3]
        my_num_layers = individual[4]
        my_nonlinearity = individual[5]
        my_leaking_rate = individual[6]
        my_spectral_radius = individual[7]
        my_w_io = individual[8]
        my_w_ih_scale = individual[9]
        my_lambda_reg = individual[10]
        my_density = individual[11]
        my_readout_training = individual[12]
        my_output_steps = individual[13]

        # add in my new function here as so

        # Training
        # trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

        print(my_nonlinearity) 

        model = nn.echo_state_network.ESN(my_input_size
             , my_hidden_size
             , my_output_size
             , num_layers = my_num_layers
             , nonlinearity = my_nonlinearity
             , batch_first = my_batch_first
             , leaking_rate = my_leaking_rate
             , spectral_radius = my_spectral_radius
             , w_io = my_w_io
             , w_ih_scale = my_w_ih_scale
             , lambda_reg = my_lambda_reg
             , density = my_density
             , readout_training = my_readout_training
             , output_steps = my_output_steps
         )

        model.to(device)

        model(trX, washout, None, trY_flat)
        model.fit()
        output, hidden = model(trX, washout)

        train_mse = loss_fcn(output, trY[washout[0]:]).item()

        # Test
        output, hidden=model(tsX, [0], hidden)
        mse=loss_fcn(output, tsY).item()
        duration=time.time() - start

        return mse, duration
          
    def mutate(individual):
        print('I am mutating!')
        gene=random.randint(3, 14)  # select which parameter to mutate
        if gene == 3:       # [3] is hidden size
            # grow or shrink the resevoir on mutate
           individual[3]=random.int(search_hidden_size_low, search_hidden_size_high)
        elif gene == 4:     # 4 number of layers
           individual[1]=random.choice(search_num_layers)

        elif gene == 5:     # 5 nonlinearity
           individual[5]=random.choice(search_nonlinearity)

        elif gene == 6:     # 6 leaking_rate
           individual[6]=random.uniform(search_leaking_rate_low,search_leaking_rate_high)

        elif gene == 7:      # 7 spectral_radius
           individual[7]=random.uniform(search_spectral_radius_low, search_spectral_radius_high)

        elif gene == 8:      # 8
           individual[8]=random.choice(search_w_io)

        elif gene == 9:     # 9 w_ih_scale
           individual[9]=random.uniform(search_w_ih_scale_low, search_w_ih_scale_high)

        elif gene == 10:      # 10 lambda ridge regression
            individual[10]=random.uniform(search_lambda_reg_low, search_lambda_reg_high)

        elif gene == 11:      # 11 density
            individual[11]=random.uniform(search_density_low, search_density_high)

        elif gene == 12:     # 12 readout_training
            individual[12]=random.choice(search_readout_training)

        elif gene == 13:     # 13 output_steps
            individual[13]=random.choice(search_output_steps)

        return individual,
        # note the final comma, leave it in the return

    ##########################################################################
    # Configure DEAP by creating a tool box and registering our configurations
    ##########################################################################

    def defineSearch(input_file_uri
            , input_size
            , output_size
            , batch_first
            , search_hidden_size_low=50
            , search_hidden_size_high=500
            , search_num_layers=[1]
            , search_nonlinearity=['tanh', 'tanh']
            , search_leaking_rate_low=1
            , search_leaking_rate_high=1
            , search_spectral_radius_low=0.1
            , search_spectral_radius_high=2
            , search_w_io=[True]
            , search_w_ih_scale_low=.5
            , search_w_ih_scale_high=2
            , search_lambda_reg_low=0.1
            , search_lambda_reg_high=1.0
            , search_density_low=0.2
            , search_density_high=1
            , search_readout_training=['cholesky', 'svd']
            , search_output_steps=['mean', 'last', 'all']
            , population_size=20
            , number_of_generations=10
            , washout=500):
        
          # create our dataset so we can reference it
          # use file handle to prep data:
          # define your univariate timeseries data set/config here
          washout = [500]
          # make our dataset global, so everything can use it
          #global trX, trY, tsX, tsY, trY_flat

          # define your dataset as so:

          device = torch.device('cuda')
          dtype = torch.double
          torch.set_default_dtype(dtype)


          if dtype == torch.double:
               data = np.loadtxt(input_file_uri, delimiter=',', dtype=np.float64)
          elif dtype == torch.float:
               data = np.loadtxt(input_file_uri, delimiter=',', dtype=np.float32)

          X_data = np.expand_dims(data[:, [0]], axis=1)
          Y_data = np.expand_dims(data[:, [1]], axis=1)
          X_data = torch.from_numpy(X_data).to(device)
          Y_data = torch.from_numpy(Y_data).to(device)

          trX = X_data[:5000]
          trY = Y_data[:5000]
          tsX = X_data[5000:]
          tsY = Y_data[5000:]

          trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)
          # trX, trY, tsX, tsY, trY_flat = trX, trY, tsX, tsY, trY_flat


          # Start search by setting up the DEAP genetic search fitness function, for ESN
          # MSE, lower is better.
          # Minimize the fitness function value
          creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
          creator.create("Individual", list, fitness=creator.FitnessMin)
          # create a toolbox
          toolbox=base.Toolbox()

          # define how we map ESN parameters to genes, and define how to randomly
          # construct new ESN individuals
          toolbox.register("attr_input_size", random.choice, input_size)  # there is only one choice
          toolbox.register("attr_output_size",random.choice, output_size)  # there is only one choice
          toolbox.register("attr_batch_first", random.choice, batch_first)  # there is only one choice
          toolbox.register("attr_hidden", random.randint, search_hidden_size_low, search_hidden_size_high)
          toolbox.register("attr_num_layers", random.choice, search_num_layers)
          toolbox.register("attr_nonlinearity", random.choice, search_nonlinearity)
          toolbox.register("attr_leaking_rate", random.uniform, search_leaking_rate_low, search_leaking_rate_high)
          toolbox.register("attr_spectral_radius", random.uniform, search_spectral_radius_low, search_spectral_radius_high)
          toolbox.register("attr_w_io", random.choice, search_w_io)
          toolbox.register("attr_w_ih_scale", random.uniform, search_w_ih_scale_low, search_w_ih_scale_high)
          toolbox.register("attr_lambda_reg", random.uniform, search_lambda_reg_low, search_lambda_reg_high)
          toolbox.register("attr_density", random.uniform, search_density_low, search_density_high)
          toolbox.register("attr_readout_training", random.choice, search_readout_training)
          toolbox.register("attr_output_steps", random.choice, search_output_steps)

          # This is the order in which genes will be combined to create a chromosome
          N_CYCLES=1

          toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.attr_input_size, toolbox.attr_output_size, toolbox.attr_batch_first, toolbox.attr_hidden, toolbox.attr_num_layers, toolbox.attr_nonlinearity, toolbox.attr_leaking_rate, toolbox.attr_spectral_radius, toolbox.attr_w_io, toolbox.attr_w_ih_scale, toolbox.attr_density, toolbox.attr_lambda_reg, toolbox.attr_density, toolbox.attr_readout_training, toolbox.attr_output_steps), n=N_CYCLES)
          toolbox.register("population", tools.initRepeat, list, toolbox.individual)

          # configure the genetic search parameters

          crossover_probability=0.7
          mutation_probability=0.3
          # the below is a heuristic that worked well for me in the past. About
          # 6% to 8% of the population is usually a good tournement size
          tournement_size=math.ceil(population_size * 0.07) + 1
          toolbox.register("mate", tools.cxOnePoint)
          toolbox.register("mutate", optimise.mutate)
          toolbox.register("select", tools.selTournament, tournsize=tournement_size)
          toolbox.register("evaluate", optimise.evaluate, trX, trY, tsX, tsY, trY_flat)

          # POP_SIZE = population_size
          pop = toolbox.population(n=population_size)
          pop = tools.selBest(pop, int(0.1 * len(pop))) + tools.selTournament(pop, len(pop) - int(0.1 * len(pop)), tournsize=tournement_size)
          hof = tools.HallOfFame(1)
          stats=tools.Statistics(lambda ind: ind.fitness.values)
          stats.register("avg", np.mean)
          stats.register("std", np.std)
          stats.register("min", np.min)
          stats.register("max", np.max)
          


          #   def fit(pop, toolbox, crossover_probability, stats, mutation_probability, number_of_generations, hof, trX, trY, tsX, tsY, trY_flat, ea_strategy='eaSimple'):
          
          # here map our configured model to the evolutionary strategy.
          # inspecting a param here and optionally calling parameterised eastrategy

          # note I need to pick up the configuration set in defineSearch, so name the things I need as global
          #if ea_strategy == 'eaSimple':
          pop, log=algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, stats=stats,
                               mutpb=mutation_probability, ngen=number_of_generations, halloffame=hof,
                               verbose=True)
          # below is a placeholder, so to demonstrate we can parameterise calling fit, to run one of many possible evolutionary strategies
          # else if ea_strategy ='your_strategy_defined_in_optimise_class':
          #    pop, log = your_strategy_defined_in_the_optimise_class(pop, toolbox, cxpb=crossover_probability, stats = stats,
          #                       mutpb = mutation_probability, ngen=number_of_generations, halloffame=hof,
          #                       verbose=True)

          end_time=datetime.datetime.now()

          # extract optimised parameters from hof, build a dict comprehension to
          # return them.
          opt_params={'attr_input_size': hof[0], 'attr_output_size': hof[1], 'attr_batch_first': hof[2], 'attr_hidden': hof[3], 'attr_num_layers': hof[4], 'attr_nonlinearity': hof[5], 'attr_leaking_rate': hof[6], 'attr_spectral_radius': hof[7], 'attr_w_io': hof[8], 'attr_w_ih_scale': hof[9], 'attr_lambda_reg': hof[10], 'attr_density': hof[11], 'attr_readout_training': hof[12], 'search_output_steps': hof[13], 'start_time': start_time, 'end_time': end_time, 'population': population_size, 'generations': number_of_generations, 'crossover_probability': crossover_probability, 'mutation_probability': mutation_probability, 'ea_strategy': ea_strategy}

          return opt_params
