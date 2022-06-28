from matplotlib import pyplot as plt
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import copy
import random

# utility module
from utilities import mkdir

import torch
from torch_tools import WaveformDataset, try_gpu, training_loop_branches
from torch.utils.data import DataLoader
from wavedecompnet_models import SeismogramEncoder, SeismogramDecoder, SeisSeparator

# make the output directory
model_dataset_dir = '.'
mkdir(model_dataset_dir)

model_structure = "Branch_Encoder_Decoder"
bottleneck_name = "LSTM"

# %% Read the pre-processed datasets
print("#" * 12 + " Loading data " + "#" * 12)
model_datasets = '/kuafu/yinjx/WaveDecompNet_dataset/training_datasets/training_datasets_all_snr_40_unshuffled.hdf5'
with h5py.File(model_datasets, 'r') as f:
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# 3. split to training (60%), validation (20%) and test (20%)
train_size = 0.6
test_size = 0.5
rand_seed1 = 13
rand_seed2 = 20
X_training, X_test, Y_training, Y_test = train_test_split(X_train, Y_train,
                                                          train_size=train_size, random_state=rand_seed1)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test,
                                                          test_size=test_size, random_state=rand_seed2)

# Give a fixed seed for model initialization
torch.manual_seed(99)
random.seed(0)
np.random.seed(20)

# Convert to the dataset class for Pytorch (here simply load all the data,
# but for the sake of memory, can also use WaveformDataset_h5)
training_data = WaveformDataset(X_training, Y_training)
validate_data = WaveformDataset(X_validate, Y_validate)

# The encoder-decoder model with LSTM bottleneck
bottleneck = torch.nn.LSTM(64, 32, 2, bidirectional=True,
                           batch_first=True, dtype=torch.float64)

# Give a name to the network
model_name = model_structure + "_" + bottleneck_name
print("#" * 12 + " building model " + model_name + " " + "#" * 12)

# Set up model network
bottleneck_earthquake = copy.deepcopy(bottleneck)
bottleneck_noise = copy.deepcopy(bottleneck)

encoder = SeismogramEncoder()
decoder_earthquake = SeismogramDecoder(bottleneck=bottleneck_earthquake)
decoder_noise = SeismogramDecoder(bottleneck=bottleneck_noise)

model = SeisSeparator(model_name, encoder, decoder_earthquake, decoder_noise).to(device=try_gpu())

# make the output directory to store the model information
model_dataset_dir = model_dataset_dir + '/' + model_name
mkdir(model_dataset_dir)

batch_size, epochs, lr = 128, 300, 1e-3
minimum_epochs = 30  # the minimum epochs that the training has to do
patience = 20  # patience of the early stopping

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
LR_func = lambda epoch: (epoch+1)/11 if (epoch<=10) else (0.95**epoch if (epoch <50) else 0.95**50) # with warmup
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LR_func, verbose=True)
train_iter = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validate_iter = DataLoader(validate_data, batch_size=batch_size, shuffle=False)

print("#" * 12 + " training model " + model_name + " " + "#" * 12)

model, avg_train_losses, avg_valid_losses, partial_loss = training_loop_branches(train_iter, validate_iter,
                                                                                 model, loss_fn, optimizer, scheduler,
                                                                                 epochs=epochs, patience=patience,
                                                                                 device=try_gpu(),
                                                                                 minimum_epochs=minimum_epochs)
print("Training is done!")

# %% Save the model
torch.save(model, model_dataset_dir + f'/{model_name}_Model.pth')

loss = avg_train_losses
val_loss = avg_valid_losses
# store the model training history
with h5py.File(model_dataset_dir + f'/{model_name}_Training_history.hdf5', 'w') as f:
    f.create_dataset("loss", data=loss)
    f.create_dataset("val_loss", data=val_loss)
    if model_structure == "Branch_Encoder_Decoder":
        f.create_dataset("earthquake_loss", data=partial_loss[0])
        f.create_dataset("earthquake_val_loss", data=partial_loss[1])
        f.create_dataset("noise_loss", data=partial_loss[2])
        f.create_dataset("noise_val_loss", data=partial_loss[3])

# add some model information
with h5py.File(model_dataset_dir + f'/{model_name}_Dataset_split.hdf5', 'w') as f:
    f.attrs['model_name'] = model_name
    f.attrs['train_size'] = train_size
    f.attrs['test_size'] = test_size
    f.attrs['rand_seed1'] = rand_seed1
    f.attrs['rand_seed2'] = rand_seed2

# %% Show loss evolution when training is done
plt.figure()
plt.plot(loss, 'o', label='loss')
plt.plot(val_loss, '-', label='Validation loss')

if model_structure == "Branch_Encoder_Decoder":
    loss_name_list = ['earthquake train loss', 'earthquake valid loss', 'noise train loss', 'noise valid loss']
    loss_plot_list = ['o', '', 'o', '']
    for ii in range(4):
        plt.plot(partial_loss[ii], marker=loss_plot_list[ii], label=loss_name_list[ii])

plt.legend()
plt.title(model_name)
plt.show()
plt.savefig(model_dataset_dir + f'/{model_name}_Training_history.png')
