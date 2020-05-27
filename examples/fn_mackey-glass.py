import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time

import random

device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)


# Here I found some great discussion on setting seeds and reproducible work in pytorch. Trying these out:
# https://discuss.pytorch.org/t/random-seed-initialization/7854/18

manualSeed = 1

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

if __name__ == "__main__":
    start = time.time()

    # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)


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

    # create the model parameters

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

    loss_fcn = torch.nn.MSELoss()


    model = ESN(input_size = _input_size
         , hidden_size = _hidden_size
         , output_size = _output_size
         , num_layers = _num_layers
         , nonlinearity = _nonlinearity
         , batch_first = _batch_first
         , leaking_rate = _leaking_rate
         , spectral_radius = _spectral_radius
         , w_io = _w_io
         , w_ih_scale = _w_ih_scale
         , lambda_reg = _lambda_reg
         , density = _density
         , readout_training = _readout_training
         , output_steps = _output_steps
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

    # print("timestamp, publog_train_err, publog_test_err, publog_runtime_sec, hidden_size, output_size, num_layers, nonlinearity, batch_first, leaking_rate, spectral_radius, w_io, w_ih_scale, lambda_reg, density, readout_training, output_step")
    print(start, publog_train_err, publog_test_err, publog_runtime_sec,_hidden_size, _output_size, _num_layers, _nonlinearity, _batch_first, _leaking_rate, _spectral_radius, _w_io, _w_ih_scale, _lambda_reg, _density, _readout_training, _output_steps ,sep=',') 

