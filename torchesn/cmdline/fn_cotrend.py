'''

This is a commandline tool version of DeepESN, that allows you to quickly build a model for any predicting a timeseries target from many 
multivariate inputs. The idea is to test whether adding new timeseries inputs into a single ESN improves our single prediction or not.
.
This cmdline utitily is used by Deap to spawn evaluating individuals in a popoulation, so we can use evolutionary search to find good hyperparameters.

To get help on the tool, try:

fn_cotrend --help

'''
import time
import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn.utils import prepare_target, washout_tensor
import math
import random

################# define the parameters to drive the script

import click

@click.command()

@click.option("--dataset", default="datasets/tcf_dji_gspc_1960_5dtrend_10k.csv", help="location of the csv datafile, in the examples/data dir. Default: 'datasets/tcf_dji_gspc_1960_5dtrend_10k.csv'")
@click.option("--input_size", default=1, type=int, help="the size of the input data, a univariate timeseries is 1: default 1")     
@click.option("--hidden_size", default=500, type=int, help="size of the hidden reservoir of the ESN: default 500")
@click.option("--output_size", default=1, type=int, help="size input size data, a univariate timeseries is 1: default 1")     
@click.option("--num_layers", default=1, type=int, help="number of layers in the DeepESN: default 1")
@click.option("--batch_first", default=False, type=click.BOOL, help="If True the input and output tensors are provided as (batch, seq, feature). Default: False")
@click.option("--leaking_rate", default=1, type=float,  help="the leaking rate of the ESN: default 1")
@click.option("--spectral_radius", default=0.9, type=float, help="the spectral radius to apply to the ESN")
@click.option("--nonlinearity", default='tanh', help="non-linearity to use. ['tanh'|'relu'|'id']: default: 'tanh'")
@click.option("--w_io", default=True, type=click.BOOL, help="teacher forcing is True/False, it included outputs back into inputs")
@click.option("--w_ih_scale", default=1, type=float, help="scaling factor to apply to teacher forcing inputs: default 1")
@click.option("--lambda_reg", default=0.6, type=float, help="ridge regression's shrinkage parameter. Default: 1")
@click.option("--density", default=0.7, type=float, help="the density/sparsity of the connections in the ESN: default 0,7")
@click.option('--readout_training', default='cholesky', type=click.Choice(['gd','svd','cholesky','inv'], case_sensitive=False),help="The readout's traning algorithm. Default 'cholesky'")
@click.option("--output_steps", default='all', type=click.Choice(['all', 'mean', 'last']), help="how the reservoir's output is used by ridge regression. ['all', 'mean', 'last']: default all")
@click.option("--seed", default=10, type=int, help="the manual seed to set for building the ESN: default 10")
@click.option("--device_mode", default='cuda', help="set your processing device, GPU or CPU. ['cuda', 'cpu']: default 'cuda'")
@click.option("--header", type = click.BOOL, default=False, help="a switch to print the header record to interpret the results. Default: False")
@click.option("--auto", type = click.BOOL, default=False, help="use this switch to only output the data needed by DEAP to evolve/optimise the parameters")

# ################# end of command line aurguments to this script

def executeESN(input_size, output_size, hidden_size, num_layers, batch_first, leaking_rate, spectral_radius, nonlinearity, w_io, w_ih_scale, lambda_reg, density, readout_training, output_steps, seed, device_mode, header, dataset, auto):
     device = torch.device('cuda')
     dtype = torch.double
     torch.set_default_dtype(dtype)

     # Here I found some great discussion on setting seeds and reproducible work in pytorch. Trying these out:
     # https://discuss.pytorch.org/t/random-seed-initialization/7854/18
     
     manualSeed = seed
     
     np.random.seed(manualSeed)
     random.seed(manualSeed)
     torch.manual_seed(manualSeed)
     # if you are suing GPU
     torch.cuda.manual_seed(manualSeed)
     torch.cuda.manual_seed_all(manualSeed)
     
     
     torch.backends.cudnn.enabled = False 
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True

     def _init_fn():
          np.random.seed(manualSeed)
         
     # prepare data for training 
     if dtype == torch.double:
         data = np.loadtxt(dataset, delimiter=',', dtype=np.float64)
     elif dtype == torch.float:
         data = np.loadtxt(dataset, delimiter=',', dtype=np.float32)

     # data prep stuff
     # assuming you have one target output in the last column of your csv file, others all input columns
     # find out how many inputs you have, doing this slowly, so it's very very clear.

     shp = data.shape
     rowcount = shp[0]   # the number of rows in the file
     colcount = shp[1]   # the number of columns in the file

     # here we accept the output size (the targets to learn, every thing else is input!
     input_size = colcount-output_size
     
     # take all columns not in the last one as X inputs 
     X_data = np.expand_dims(data[:, :input_size], axis=1)

     # take the last column as the Y target to predict
     Y_data = np.expand_dims(data[:, input_size:colcount], axis=1)

     X_data = torch.from_numpy(X_data).to(device)
     Y_data = torch.from_numpy(Y_data).to(device)

     train_test_split = math.ceil(rowcount/2)

     trX = X_data[:train_test_split]
     trY = Y_data[:train_test_split]
     tsX = X_data[train_test_split:]
     tsY = Y_data[train_test_split:]

     washout = [500]

     # Training
     trY_flat = prepare_target(trY.clone(), [trX.size(0)], washout)

     loss_fcn = torch.nn.MSELoss()

     # end of data prep.
     # Timer. we start timer once data is prepared. Duration is used as a measure in fn_autotune, as part of the fitness function's multi-objective
     # evolution, so that we can find high performing, efficient (fast) ESN  models, rather than just bigger models. Time includes training + testing
     # If you tailor the code to use multiple GPUs of different speeds/types etc, this might need a rethink

     start = time.time()

     # call the configured model
     model = ESN(input_size
         , hidden_size
         , output_size
         , num_layers 
         , nonlinearity
         , batch_first 
         , leaking_rate 
         , spectral_radius 
         , w_io 
         , w_ih_scale 
         , lambda_reg 
         , density 
         , readout_training 
         , output_steps 
         )

     model.to(device)

     model(trX, washout, None, trY_flat)
     model.fit()
     output, hidden = model(trX, washout)

     # here is the loss of the Train error
     publog_train_err = loss_fcn(output, trY[washout[0]:]).item()

     # here is the loss of the Test error, and the duration of the run
     output, hidden = model(tsX, [0], hidden)
     publog_test_err = loss_fcn(output, tsY).item()
     publog_runtime_sec =  time.time() - start

     if header == True:
          # print the header record, if asked for
          print("timestamp,publog_train_err,publog_test_err,publog_runtime_sec,hidden_size,output_size,seed,num_layers,nonlinearity,batch_first,leaking_rate,spectral_radius,w_io,w_ih_scale,lambda_reg,density,readout_training,output_steps,dataset")
     
     # print fitness and parameter data
     if auto == True:
          print(publog_test_err, publog_runtime_sec)
          # build a dict comprehension, for the other data we generally output, convert to json, write to fixed log file, 
          # so we capture the whole log for an entire evolutionary run if this is called from fn_autotune.
          # the statistics from the log file could be useful for analysis of the evolution itself.

     else:
          print(start, publog_train_err, publog_test_err, publog_runtime_sec, hidden_size, output_size, seed,num_layers, nonlinearity, batch_first, leaking_rate, spectral_radius, w_io, w_ih_scale, lambda_reg, density, readout_training, output_steps, dataset ,sep=',') 

if __name__ == "__main__":
     executeESN()
