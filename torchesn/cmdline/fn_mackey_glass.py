
import time
import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn.utils import prepare_target, washout_tensor

#import torch.nn
#import numpy as np
#import torchesn
# import nn
#from torchesn import utils 

import random

################# define the parameters to drive the script

import click

@click.command()

@click.option("--dataset", default="datasets/mg17.csv", help="location of the csv datafile, in the examples/data directory. Default: 'datasets/mg17.csv'")
@click.option("--input_size", default=1, help="the size of the input data, a univariate timeseries is 1: default 1")     
@click.option("--hidden_size", default=500, help="size of the hidden reservoir of the ESN: default 500")
@click.option("--output_size", default=1, help="size input size data, a univariate timeseries is 1: default 1")     
@click.option("--num_layers", default=1, help="number of layers in the DeepESN: default 1")
@click.option("--batch_first", default=False, help="If True the input and output tensors are provided as (batch, seq, feature). Default: False")
@click.option("--leaking_rate", default=1, help="the leaking rate of the ESN: default 1")
@click.option("--spectral_radius", default=0.9, help="the spectral radius to apply to the ESN")
@click.option("--nonlinearity", default='tanh', help="non-linearity to use. ['tanh'|'relu'|'id']: default: 'tanh'")
@click.option("--w_io", default=True, help="teacher forcing is True/False, it included outputs back into inputs")
@click.option("--w_ih_scale", default=1, help="scaling factor to apply to teacher forcing inputs: default 1")
@click.option("--lambda_reg", default=0.6, help="ridge regression's shrinkage parameter. Default: 1")
@click.option("--density", default=0.7, help="the density/sparsity of the connections in the ESN: default 0,7")
@click.option('--readout_training', default='cholesky', type=click.Choice(['gd','svd','cholesky','inv'], case_sensitive=False),help="The readout's traning algorithm. Default 'cholesky'")
@click.option("--output_steps", default='all', help="how the reservoir's output is used by ridge regression. ['all', 'mean', 'last']: default all")
@click.option("--seed", default=10, help="the manual seed to set for building the ESN: default 10")
@click.option("--device_mode", default='cuda', help="set your processing device, GPU or CPU. ['cuda', 'cpu']: default 'cuda'")
@click.option("--header", type = click.BOOL, default=False, help="a switch to print the header record to interpret the results. Default: False")
# ################# end of command line aurguments to this script

def executeESN(input_size, output_size, hidden_size, num_layers, batch_first, leaking_rate, spectral_radius, nonlinearity, w_io, w_ih_scale, lambda_reg, density, readout_training, output_steps, seed, device_mode, header, dataset):
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
         
 
     if dtype == torch.double:
         data = np.loadtxt(dataset, delimiter=',', dtype=np.float64)
     elif dtype == torch.float:
         data = np.loadtxt(dataset, delimiter=',', dtype=np.float32)
     X_data = np.expand_dims(data[:, [0]], axis=1)
     Y_data = np.expand_dims(data[:, [1]], axis=1)
     X_data = torch.from_numpy(X_data).to(device)
     Y_data = torch.from_numpy(Y_data).to(device)

     trX = X_data[:5000]
     trY = Y_data[:5000]
     tsX = X_data[5000:]
     tsY = Y_data[5000:]

     washout = [500]

     start = time.time()

     # Training
     trY_flat = prepare_target(trY.clone(), [trX.size(0)], washout)


     loss_fcn = torch.nn.MSELoss()

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

     publog_train_err = loss_fcn(output, trY[washout[0]:]).item()

     # Test
     output, hidden = model(tsX, [0], hidden)
     publog_test_err = loss_fcn(output, tsY).item()
     publog_runtime_sec =  time.time() - start

     if header == True:
          # print the header record, if asked for
          print("timestamp, publog_train_err, publog_test_err, publog_runtime_sec, hidden_size, output_size, seed, num_layers, nonlinearity, batch_first, leaking_rate, spectral_radius, w_io, w_ih_scale, lambda_reg, density, readout_training, output_steps, dataset")
     
     # print fitness and parameter data
     print(start, publog_train_err, publog_test_err, publog_runtime_sec, hidden_size, output_size, num_layers, seed, nonlinearity, batch_first, leaking_rate, spectral_radius, w_io, w_ih_scale, lambda_reg, density, readout_training, output_steps, dataset ,sep=',') 

    

if __name__ == "__main__":
     executeESN()
