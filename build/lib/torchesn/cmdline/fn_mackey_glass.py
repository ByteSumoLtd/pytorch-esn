
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

@click.option("--input_size", default=1, help="size input size data, a univariate timeseries is 1: default 1")     
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

def executeESN(input_size, output_size, hidden_size, num_layers, batch_first, leaking_rate, spectral_radius, nonlinearity, w_io, w_ih_scale, lambda_reg, density, readout_training, output_steps, seed, device_mode, header):
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
         data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float64)
     elif dtype == torch.float:
         data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float32)
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


     # Arguments to test a range of ESN options
     #Args:
     #    input_size: The number of expected features in the input x.
     #    hidden_size: The number of features in the hidden state h.
     #    output_size: The number of expected features in the output y.
     #    num_layers: Number of recurrent layers. Default: 1
     #    nonlinearity: The non-linearity to use ['tanh'|'relu'|'id'].
     #        Default: 'tanh'
     #    batch_first: If ``True``, then the input and output tensors are provided
     #        as (batch, seq, feature). Default: ``False``
     #    leaking_rate: Leaking rate of reservoir's neurons. Default: 1
     #    spectral_radius: Desired spectral radius of recurrent weight matrix.
     #        Default: 0.9
     #    w_ih_scale: Scale factor for first layer's input weights (w_ih_l0). It
     #        can be a number or a tensor of size '1 + input_size' and first element
     #        is the bias' scale factor. Default: 1
     #    lambda_reg: Ridge regression's shrinkage parameter. Default: 1
     #    density: Recurrent weight matrix's density. Default: 1
     #    w_io: If 'True', then the network uses trainable input-to-output
     #        connections. Default: ``False``
     #    readout_training: Readout's traning algorithm ['gd'|'svd'|'cholesky'|'inv'].
     #        If 'gd', gradients are accumulated during backward
     #        pass. If 'svd', 'cholesky' or 'inv', the network will learn readout's
     #        parameters during the forward pass using ridge regression. The
     #        coefficients are computed using SVD, Cholesky decomposition or
     #        standard ridge regression formula. 'gd', 'cholesky' and 'inv'
     #        permit the usage of mini-batches to train the readout.
     #        If 'inv' and matrix is singular, pseudoinverse is used.
     #    output_steps: defines how the reservoir's output will be used by ridge
     #        regression method ['all', 'mean', 'last'].
     #        If 'all', the entire reservoir output matrix will be used.
     #        If 'mean', the mean of reservoir output matrix along the timesteps
     #        dimension will be used.
     #        If 'last', only the last timestep of the reservoir output matrix
     #        will be used.
     #        'mean' and 'last' are useful for classification tasks.

     '''
         # create the model parameters
         # this was the old param setting, reverting to @click on commandline
         _input_size = 1 
         _output_size = 1
         _hidden_size = 2000
         _num_layers = 1
         _nonlinearity = 'tanh'
         _batch_first = False
         _leaking_rate = 1
         _spectral_radius = 1.7
         _w_io = True
         _w_ih_scale = 1
         _lambda_reg = 0.6
         _density = 0.7
         _readout_training = 'cholesky'
         _output_steps = 'all'
      '''
     loss_fcn = torch.nn.MSELoss()


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

     #print("Training error:", loss_fcn(output, trY[washout[0]:]).item())
     publog_train_err = loss_fcn(output, trY[washout[0]:]).item()

     # Test
     output, hidden = model(tsX, [0], hidden)
     #print("Test error:", loss_fcn(output, tsY).item())
     publog_test_err = loss_fcn(output, tsY).item()

     #print("Ended in", time.time() - start, "seconds.")
     publog_runtime_sec =  time.time() - start

     if header == True:
          # print the header record, if asked for
          print("timestamp, publog_train_err, publog_test_err, publog_runtime_sec, hidden_size, output_size, seed, num_layers, nonlinearity, batch_first, leaking_rate, spectral_radius, w_io, w_ih_scale, lambda_reg, density, readout_training, output_steps")
     # print fitness and parameter data
     print(start, publog_train_err, publog_test_err, publog_runtime_sec, hidden_size, output_size, num_layers, seed, nonlinearity, batch_first, leaking_rate, spectral_radius, w_io, w_ih_scale, lambda_reg, density, readout_training, output_steps ,sep=',') 

    

if __name__ == "__main__":
     executeESN()
