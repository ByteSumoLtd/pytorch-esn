import torch.nn
from torchvision import datasets, transforms
from torchesn.nn import ESN
import time


def Accuracy_Correct(y_pred, y_true):
    labels = torch.argmax(y_pred, 1).type(y_pred.type())
    correct = len((labels == y_true).nonzero())
    return correct


def one_hot(y, output_dim):
    onehot = torch.zeros(y.size(0), output_dim, device=y.device)

    for i in range(output_dim):
        onehot[y == i, i] = 1

    return onehot


def reshape_batch(batch):
    batch = batch.view(batch.size(0), batch.size(1), -1)
    return batch.transpose(0, 1).transpose(0, 2)


device = torch.device('cuda')
dtype = torch.float
torch.set_default_dtype(dtype)
loss_fcn = Accuracy_Correct

batch_size = 712  # Tune it according to your VRAM's size.
washout_rate = 0.2

# create the model parameters

input_size = 1
output_size = 10
hidden_size = 30
num_layers = 5
nonlinearity = 'tanh'
batch_first = False
leaking_rate = 1
spectral_radius = 0.9
w_ih_scale = 1
lambda_reg = 1
density = 1
readout_training = 'svd'
output_steps = 'mean'

if __name__ == "__main__":
    train_iter = torch.utils.data.DataLoader(
        datasets.MNIST('./datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    test_iter = torch.utils.data.DataLoader(
        datasets.MNIST('./datasets', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    start = time.time()

    # Training
    model = ESN(input_size = input_size
         , hidden_size = hidden_size
         , output_size = output_size
         , num_layers = num_layers
         , nonlinearity = nonlinearity
         , batch_first = batch_first
         , leaking_rate = leaking_rate
         , spectral_radius = spectral_radius
         , w_ih_scale = w_ih_scale
         , lambda_reg = lambda_reg
         , density = density
         , readout_training = readout_training
         , output_steps = output_steps
         )

    model.to(device)

    # Fit the model
    for batch in train_iter:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        x = reshape_batch(x)
        target = one_hot(y, output_size)
        washout_list = [int(washout_rate * x.size(0))] * x.size(1)

        model(x, washout_list, None, target)
        model.fit()

    # Evaluate on training set
    tot_correct = 0
    tot_obs = 0

    for batch in train_iter:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        x = reshape_batch(x)
        washout_list = [int(washout_rate * x.size(0))] * x.size(1)

        output, hidden = model(x, washout_list)
        tot_obs += x.size(1)
        tot_correct += loss_fcn(output[-1], y.type(torch.get_default_dtype()))

    print("Training accuracy:", tot_correct / tot_obs)

    # Test
    for batch in test_iter:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        x = reshape_batch(x)
        washout_list = [int(washout_rate * x.size(0))] * x.size(1)

        output, hidden = model(x, washout_list)
        tot_obs += x.size(1)
        tot_correct += loss_fcn(output[-1], y.type(torch.get_default_dtype()))

    print("Test accuracy:", tot_correct / tot_obs)

    print("Ended in", time.time() - start, "seconds.")
