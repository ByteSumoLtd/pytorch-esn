'''
This module is a Utility library to help you use genetic algorithms to tune up your ESN / DeepESN by finding you good hyperparameters.
It is based on DEAP (Distributed Evolutionary Algorithms in Python) using a pattern delivering basic genetic search.

This library should not be used directly, but called for using the command line tool.
For the moment, the command line tool will work for any univariate timeseries problem you configure as input having two cols, [Observation],[Target].

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
import time
import subprocess
import os
import pprint
import multiprocessing


# Define how to call and evaluate the pytorch-esn function as an genetic
# Individual, where hyperparamters are considered genes.
def evaluate(individual):
     '''
        Evaluate the individual proposed by evolution for fitness, via the simplified cmdline tool interface.
        This function marshalls in all the parameters, calls out to the cmdline tool, and fetches the returned fitness information 
        To tailor the genetic search for your particular problem and dataset, duplicate 'fn_mackey_glass.py', and customise it, then save and
        rename it, ensuring it can run from the examples directory. Then reinstall to activate it. 
        Once it's ready to go, you can call "fn_autotune" with the name of your new cmdline function as a parameter, so it's executed here and is
        finding optimal parameters for your own data pipeline.
     '''
     # make sure we are working in the examples directory, from where we can call the cmdline tools.
     os.chdir('/home/andrew/pytorch-esn/examples')




     # extract the values for the parameters from the individual chromosome
     my_input_size = individual[0]     #ok
     my_output_size = individual[1]    #ok
     my_batch_first = individual[2]    #ok
     my_hidden_size = individual[3]    #ok
     my_num_layers = individual[4]     #ok
     my_nonlinearity = individual[5]   #ok
     my_leaking_rate = individual[6]   #ok
     my_spectral_radius = individual[7] #ok
     my_w_io = individual[8]            #ok
     my_w_ih_scale = individual[9]      #ok
     my_density = individual[10]        #ok
     my_lambda_reg = individual[11]     #ok
     my_readout_training = individual[12] #ok
     my_output_steps = individual[13]     #ok
     my_cmdline_tool = individual[14]     #ok
     my_dataset =  individual[15]         # for now only fn_mackey_glass is configured

     # construct the command line to run the individual using the cmdline fn
     runstring = my_cmdline_tool + " --dataset " + my_dataset + " --input_size " + str(my_input_size) + " --output_size " + str(my_output_size) + " --batch_first " + str(my_batch_first) + " --hidden_size " + str(my_hidden_size) + " --num_layers " + str(my_num_layers) + " --leaking_rate " + str(my_leaking_rate) + " --spectral_radius " + str(my_spectral_radius) + " --nonlinearity " + str(my_nonlinearity) + " --w_io " + str(my_w_io) + " --w_ih_scale " + str(my_w_ih_scale) + " --lambda_reg " + str(my_lambda_reg) + " --density " + str(my_density) + " --readout_training " + str(my_readout_training) + " --output_steps " + str(my_output_steps) + " --auto true"

     #result = subprocess.run(runstring, stdout=subprocess.PIPE)

     stream = os.popen(runstring)
     result = stream.read()
 
     # add in my new function here as so
     # Test

     test_mse = float(result.split(" ")[0])
     duration = float(result.split(" ")[1])
     #print(runstring, " #  test_error: ", test_mse," # eval_duration: ", duration)    
     return test_mse, duration
          

##########################################################################
# Configure DEAP by creating a tool box and registering our configurations
##########################################################################

def defineSearch(
              input_file_uri
            , input_size
            , output_size
            , batch_first
            , population_size=30
            , number_of_generations=15
            , search_hidden_size_low=100
            , search_hidden_size_high=1000
            , search_min_num_layers=1
            , search_max_num_layers=2
            , search_nonlinearity=['tanh', 'tanh']
            , search_leaking_rate_low=0.1
            , search_leaking_rate_high=1.0
            , search_spectral_radius_low=0.1
            , search_spectral_radius_high=2
            , search_w_io=[True, False, True, True]
            , search_w_ih_scale_low=0.4
            , search_w_ih_scale_high=1.6
            , search_lambda_reg_low=0.1
            , search_lambda_reg_high=1.0
            , search_density_low=0.2
            , search_density_high=1.0
            , search_readout_training=['cholesky', 'svd', 'cholesky', 'cholesky', 'cholesky']
            , search_output_steps=['mean', 'last', 'all', 'all', 'all', 'all']
            , washout=500
            , cmdline_tool='fn_mackey_glass'
            , pool_size=1
            , number_islands=0 # if we get this parameter >0 then we'll allocate the population_size to each island
            , ):

      # Reproduciple evolution:
      # I noticed you can force DEAP to use a standard seed, to make runs reproducible I think? Not sure if that's good overall
      # as a user requesting a second run would want a second opinion for ESN settings? 
      # however, I'm putting in a placeholder here for it, as it could be thing to explore...
      # Uncomment the below to trial the manual seed setting functionality:
      # random.seed(64)

      # Private functions used by DEAP to run the evolution:
      # define our bespoke mutate function, that needs the default params set above
      def mutate(individual):
      # print('I am mutating!')
          gene=random.randint(3, 14)  # select which parameter to mutate
          if gene == 3:       # [3] is hidden size
              # grow or shrink the resevoir on mutate
              individual[3]= random.randint(search_hidden_size_low, search_hidden_size_high)

          elif gene == 4:     # 4 number of layers
              individual[1]= random.randint(search_min_num_layers, search_max_num_layers)

          elif gene == 5:     # 5 nonlinearity
              individual[5]=random.choice(search_nonlinearity)

          elif gene == 6:     # 6 leaking_rate
              individual[6]=random.uniform(search_leaking_rate_low,search_leaking_rate_high)

          elif gene == 7:      # 7 spectral_radius
              individual[7]=random.uniform(search_spectral_radius_low, search_spectral_radius_high)

          elif gene == 8:      # 8 w_io
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

      # Start configuring out DEAP search by setting up the DEAP genetic search fitness function, for ESN
      # For our problems minimimising MSE, lower fitness scores is better. Shorter runtimes also preferable to long running ones.
      # So we will set two fitness scores, MSE and Duration and blend with weights the search... to find good and efficient ESNs architectures.

      creator.create("FitnessMulti", base.Fitness, weights=(-1.0000, -0.10)) 
      # above we set out mse, and runtime, in that order for fitness.  
      # I'm weighting runtime duration lower, at 10%, as I'm primarily interested in best solutions, and want tie breakers prefering lowest cost.
      creator.create("Individual", list, fitness=creator.FitnessMulti)


      # create a toolbox for deap.
      toolbox=base.Toolbox()

      # define how we map ESN parameters to genes, and define how to randomly
      # construct new ESN "individuals" representing ESNs with different hyperparameters
      toolbox.register("attr_input_size", random.choice, [input_size])  # there is only one choice
      toolbox.register("attr_output_size",random.choice, [output_size])  # there is only one choice
      toolbox.register("attr_batch_first", random.choice, [batch_first])  # there is only one choice
      toolbox.register("attr_hidden", random.randint, search_hidden_size_low, search_hidden_size_high)
      toolbox.register("attr_num_layers", random.randint, search_min_num_layers, search_max_num_layers)
      toolbox.register("attr_nonlinearity", random.choice, search_nonlinearity)
      toolbox.register("attr_leaking_rate", random.uniform, search_leaking_rate_low, search_leaking_rate_high)
      toolbox.register("attr_spectral_radius", random.uniform, search_spectral_radius_low, search_spectral_radius_high)
      toolbox.register("attr_w_io", random.choice, search_w_io)
      toolbox.register("attr_w_ih_scale", random.uniform, search_w_ih_scale_low, search_w_ih_scale_high)
      toolbox.register("attr_lambda_reg", random.uniform, search_lambda_reg_low, search_lambda_reg_high)
      toolbox.register("attr_density", random.uniform, search_density_low, search_density_high)
      toolbox.register("attr_readout_training", random.choice, search_readout_training)
      toolbox.register("attr_output_steps", random.choice, search_output_steps)
      toolbox.register("attr_cmdline_tool", random.choice, [cmdline_tool])  # note, this is just a single value, but we need it to eval the individual
      toolbox.register("attr_dataset", random.choice, [input_file_uri])     # note, this is just a single value, but we need it to eval the individual
      # This is the order in which genes will be combined to create a chromosome
      N_CYCLES=1

      toolbox.register("individual", tools.initCycle, creator.Individual,(
		toolbox.attr_input_size         #0 #genes are positional, the idx mapped in comments is critical to get correct elsewhere 
                , toolbox.attr_output_size      #1
                , toolbox.attr_batch_first      #2
                , toolbox.attr_hidden           #3
                , toolbox.attr_num_layers       #4
                , toolbox.attr_nonlinearity     #5
                , toolbox.attr_leaking_rate     #6
                , toolbox.attr_spectral_radius  #7
                , toolbox.attr_w_io             #8
                , toolbox.attr_w_ih_scale       #9
                , toolbox.attr_density          #10
                , toolbox.attr_lambda_reg       #11
                , toolbox.attr_readout_training #12
                , toolbox.attr_output_steps     #13
                , toolbox.attr_cmdline_tool     #14
                , toolbox.attr_dataset)         #15
                , n=N_CYCLES)
      # mental note: the above index numbers are needed to access these variables from hof, the hall_of_fame best individual evolved by our process
      # hof includes the whole population, so we set the best_params = hof[0] and then access values like this:  best_nonlinearity = best_params[5]

      # this creates the random initial population of ESN solutions to evolve, coded in genes 
      toolbox.register("population", tools.initRepeat, list, toolbox.individual)

      # configure the genetic search parameters. We can tune these, but these are fairly standard defaults
      crossover_probability=0.7
      mutation_probability=0.3

      # Dynamic Settings

      # the below is a heuristic that worked well for me in the past. About
      # 6% to 8% of the population is usually a good tournement size (from experience with Karoo_gp)
      tournement_size=math.ceil(population_size * 0.07) + 1
      # as the tournement heuristic works, why not set the migrants between islands parameter to the same dynamic value?
      k_migrants = tournement_size
      # here, we calc from total number of generations, a FREQ value that implies 5ish deme migrations per run
      FREQ = math.ceil(number_of_generations * 0.21) + 1
      
      # set out evolution functions. Note we defined a bespoke mutate function, as our hyperparams/genes have mixed types.
      toolbox.register("mate", tools.cxOnePoint)
      toolbox.register("mutate", mutate)
      toolbox.register("select", tools.selTournament, tournsize=tournement_size)
      toolbox.register("evaluate", evaluate)

      # POP_SIZE = population_size
      # moving the below population management into conditional block below, to offer deme / no deme options
      # pop = toolbox.population(n=population_size)
      # pop = tools.selBest(pop, int(0.1 * len(pop))) + tools.selTournament(pop, len(pop) - int(0.1 * len(pop)), tournsize=tournement_size)

      # set out administration functions for deap, hall of fame (best) and statistics
      hof = tools.HallOfFame(1)
      stats=tools.Statistics(lambda ind: ind.fitness.values)
      stats.register("avg", np.mean)
      stats.register("std", np.std)
      stats.register("min", np.min)
      stats.register("max", np.max)

      # set up a pool of evaluation workers here. Be sure to check your GPU can handle concurrent evals.
      # later I will set up a worker_pool_size as a command line parameter to fn_autotune

      # Process Pool of 4 workers, the size of the pool comes from the commandline now, or defaults to 1
      pool = multiprocessing.Pool(processes=pool_size)
      toolbox.register("map", pool.map)

      # just before we start, lets grab the start time, and calc a duration
      start_time = datetime.datetime.now()
      print(start_time)
      # this command runs the genetic evolution - and it may take several hours, depending on your params          
      

      ##################### Evolve Solution for a single population:
      if number_islands == 0:
          # POP_SIZE = population_size
          pop = toolbox.population(n=population_size)
          pop = tools.selBest(pop, int(0.1 * len(pop))) + tools.selTournament(pop, len(pop) - int(0.1 * len(pop)), tournsize=tournement_size)

          # run simple single population evolution 
          pop, log=algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, stats=stats,
                               mutpb=mutation_probability, ngen=number_of_generations, halloffame=hof,
                               verbose=True)

      ##################### Evolve "multidemic" Solution having subpopulations, with "ring migrations" occuring on each FREQ defined generation
      elif number_islands > 0:
          # based on an example herre: https://github.com/DEAP/deap/blob/master/examples/ga/onemax_multidemic.py 
          # note:our user parameters called number_islands = number_demes

          # define island ring migration strategy: we only migrate unique individuals, hence "replacement=random.sample" 
          toolbox.register("migrate", tools.migRing, k=k_migrants, selection=tools.selBest, replacement=random.sample)

          # define demes, which are sub-populations, or "islands" of size population_size. Note global population = number_islands*population_size
          demes = [toolbox.population(n=population_size) for _ in range(number_islands)]

          # add extra logging for demes
          log = tools.Logbook()
          log.header = "gen", "deme", "evals", "std", "min", "avg", "max"
 
          # configure demes: define fitness within deme, stats, hof, and logging
          for idx, deme in enumerate(demes):
              #for ind in deme:
                  #ind.fitness.values = toolbox.evaluate(ind)  # no parallel run 

              demewide_ind = [ind for ind in deme]
              fitnesses = toolbox.map(toolbox.evaluate, demewide_ind)
              for ind, fit in zip(demewide_ind, fitnesses):
                  ind.fitness.values = fit


              #stats.update(deme, idx)
              log.record(gen=0, deme=idx, evals=len(deme), **stats.compile(deme))
              hof.update(deme)
              # debug / outputs to pty, can comment out
              print(log.stream)

          # test: create a little function to run an individual's fitness eval, to simplify parallelism
          #def queueEval(ind):
          #    ind.fitness.values = toolbox.evaluate(ind)
          #    return

          # Run Deme based evolution
          gen = 1
          while gen <= number_of_generations and log[-1]["min"] > 0:     # halt if MSE "min" is zero, we've solved the problem

              for idx, deme in enumerate(demes):
                  deme[:] = toolbox.select(deme, len(deme))
                  deme[:] = algorithms.varAnd(deme, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)

                  invalid_ind = [ind for ind in deme if not ind.fitness.valid]

                  #for ind in invalid_ind:    
                  #    ind.fitness.values = toolbox.evaluate(ind)  # original code

                  # the below evaluates fitness in parallel. good!
                  fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                  for ind, fit in zip(invalid_ind, fitnesses):
                      ind.fitness.values = fit

                  log.record(gen=gen, deme=idx, evals=len(deme), **stats.compile(deme))
                  hof.update(deme)
              print(log.stream)

              # On a pulse of FREQ, force ring migration of individuals across our demes/islands
              if gen % FREQ == 0:
                  toolbox.migrate(demes)
                  print("------------------------migration across islands--------------")
              gen += 1


      # do house keeping to close our pool, and record ending timestampt
      pool.close()
      end_time = datetime.datetime.now()

      # debug / reporting: pass back endtime in readable format to pty, comment out if unwanted
      print(end_time)

      # debug: we can pretty print hof to pty, versus outputs printed by caller if checking for a mismatch/bug
      # pp = pprint.PrettyPrinter(indent=4)  
      # pprint(hof[0])

      # assign best found individual across the hall of fame as our output, best_params.
      best_params = hof[0]
      best_fitness = hof[0].fitness

      # create final output:
      # build a dict comprehension, to collect all the best parameters of the ESN, and settings used to find it
      # is needed to so  we can return interpretable results of evolution back to the user/caller function
      opt_params={'run_training_loss': best_fitness ,'attr_input_size': best_params[0], 'attr_output_size': best_params[1], 'attr_batch_first': best_params[2], 'attr_hidden': best_params[3], 'attr_num_layers': best_params[4], 'attr_nonlinearity': best_params[5], 'attr_leaking_rate': best_params[6], 'attr_spectral_radius': best_params[7], 'attr_w_io': best_params[8], 'attr_w_ih_scale': best_params[9], 'attr_density': best_params[10],'attr_lambda_reg': best_params[11], 'attr_readout_training': best_params[12], 'attr_output_steps': best_params[13], '_cmdline_tool': cmdline_tool, 'run_start_time': start_time, 'run_end_time': end_time, 'auto_population': population_size, 'auto_islands': number_islands, 'auto_generations': number_of_generations, 'auto_crossover_probability': crossover_probability, 'auto_mutation_probability': mutation_probability}

      # pass back our combined record of results to caller
      return opt_params
