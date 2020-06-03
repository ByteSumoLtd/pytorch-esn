# PyTorch-ESN, with Genetic Algorithms for Tuning

PyTorch-ESN is a PyTorch module, written in Python, implementing Echo State Networks with leaky-integrated units. ESN's implementation with more than one layer is based on [DeepESN](https://arxiv.org/abs/1712.04323). The readout is trainable by ridge regression or by PyTorch's optimizers.

Its development started under Stefano's master thesis titled "An Empirical Comparison of Recurrent Neural Networks on Sequence Modeling", which was supervised by Prof. Alessio Micheli and Dr. Claudio Gallicchio at the University of Pisa.

This https://github.com/ByteSumoLtd/pytorch-esn fork of Stefano's core library, and adds the following

- examples of running a grid search of mackey-glass hyperparameters, illustrates hyperparams have a large effect on performance
- conversion of univariate mackey-glass example, into a general command line tool for any univariate timeseries having two csv columns 'Input, Target'. The file should not have a header.
- ability to manually set a seed in the commandline tool, for reproducible results
- some hacky helpers to reinstall the modules when in active development
- inclusion of torchesn/optim and an optimise module that configures DEAP to search for good ESN hyperparamets. Opinionated but with good defaults.
- inclusion of fn_autotune commandline tool, that automates discovery of good hyper-parameters for your problem, automating the whole ESN training process
- inclusion of multiprocessing as a parallisation mechanism to accelerate genetic search for good models on a single GPU, and parameters to simply set worker number


to do:
- include a python notebook for a worked example, illustrating the cost / benefits of genetic search

Examples:
a) in the examples directory, test executing the mackey-glass example, using some parameters: 

<pre><code> fn_mackey_glass --hidden_size 15 --input_size 1 --output_size 1 --spectral_radius 0.8981482392105415 --density 0.5411114043211104 --leaking_rate 0.31832532215823006 --lambda_reg 0.3881908755655027 --nonlinearity tanh --readout_training cholesky --w_io True --w_ih_scale 0.826708317582006 --seed 10
</code></pre>

B) in the examples directory, test using genetic search to find good hyperparameters for mackey-glass:

<pre><code>
> fn_autotune --population 30 --generations 10 --max_layers 1 --hidden_size_low 150 --hidden_size_high 150 --worker_pool 6 | tee logs/example.log 
> cat logs/example.log

2020-06-03 10:37:48.642212
gen	nevals	avg    	std    	min        	max    
0  	30    	2.83442	2.86397	2.09471e-05	6.65214
1  	20    	2.78962	2.84464	2.09471e-05	7.11487
2  	24    	2.76368	2.80678	1.33755e-05	6.85562
3  	22    	3.17741	3.22544	1.33755e-05	7.29831
4  	26    	3.09174	3.12538	1.33755e-05	7.23457
5  	25    	2.95496	3.02999	1.13259e-05	7.81776
6  	24    	3.25476	3.33798	1.13259e-05	7.61532
7  	23    	2.86864	2.95118	1.13259e-05	7.61532
8  	25    	2.81973	2.87699	1.13259e-05	7.12179
9  	27    	2.82086	2.86294	1.13259e-05	7.34068
10 	25    	2.98786	3.06309	1.13259e-05	7.82064
2020-06-03 10:46:13.766080
{   '_cmdline_tool': 'fn_mackey_glass',
    'attr_batch_first': False,
    'attr_density': 0.6905894761994907,
    'attr_hidden': 150,
    'attr_input_size': 1,
    'attr_lambda_reg': 0.6304891830771884,
    'attr_leaking_rate': 0.9440340390508313,
    'attr_nonlinearity': 'tanh',
    'attr_num_layers': 1,
    'attr_output_size': 1,
    'attr_output_steps': 'all',
    'attr_readout_training': 'cholesky',
    'attr_spectral_radius': 1.3467750025214633,
    'attr_w_ih_scale': 0.43537928601439935,
    'attr_w_io': True,
    'auto_crossover_probability': 0.7,
    'auto_generations': 10,
    'auto_mutation_probability': 0.3,
    'auto_population': 30,
    'run_end_time': datetime.datetime(2020, 6, 3, 10, 46, 13, 766080),
    'run_start_time': datetime.datetime(2020, 6, 3, 10, 37, 48, 642212),
    'run_training_loss': deap.creator.FitnessMulti((1.1325887772018666e-05, 4.748876571655273))}

# the command line view of the params is:
# fn_mackey_glass --hidden_size 150 --input_size 1 --output_size 1 --spectral_radius 1.3467750025214633 --density 0.6905894761994907 --leaking_rate 0.9440340390508313 --lambda_reg 0.6304891830771884 --nonlinearity tanh --readout_training cholesky --w_io True --w_ih_scale 0.43537928601439935 --seed 10

</code></pre>





## Prerequisites

* PyTorch, deap, multiprocessing, pprint, click 

## Basic Usage

### Offline training (ridge regression)

#### SVD
Mini-batch mode is not allowed with this method.

```python
from torchesn.nn import ESN
from torchesn.utils import prepare_target

# prepare target matrix for offline training
flat_target = prepare_target(target, seq_lengths, washout)

model = ESN(input_size, hidden_size, output_size)

# train
model(input, washout, hidden, flat_target)

# inference
output, hidden = model(input, washout, hidden)
```

#### Cholesky or inverse
```python
from torchesn.nn import ESN
from torchesn.utils import prepare_target

# prepare target matrix for offline training
flat_target = prepare_target(target, seq_lengths, washout)

model = ESN(input_size, hidden_size, output_size, readout_training='cholesky')

# accumulate matrices for ridge regression
for batch in batch_iter:
    model(batch, washout[batch], hidden, flat_target)

# train
model.fit()

# inference
output, hidden = model(input, washout, hidden)
```

#### Classification tasks
For classification, just use one of the previous methods and pass 'mean' or
'last' to ```output_steps``` argument.

```python
model = ESN(input_size, hidden_size, output_size, output_steps='mean')
```

For more information see docstrings or section 4.7 of "A Practical Guide to Applying
Echo State Networks" by Mantas Lukoševičius.

### Online training (PyTorch optimizer)

Same as PyTorch.
