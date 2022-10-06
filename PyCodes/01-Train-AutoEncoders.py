# GPU checks
import tensorflow as tf

tf.test.gpu_device_name()

# GPU checks
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from importlib import reload
from helper_funcs import *
from make_models2 import *

# Make tqdm work for notebooks
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

import os
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import pickle
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# Run GPU test
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print(
        '\n\nThis error most likely means that this notebook is not '
        'configured to use a GPU.  Change this in Notebook Settings via the '
        'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
    raise SystemError('GPU device not found')

# LOAD Data
arr = np.load("/neurospin/psy_sbox/rl264746/ABIDE-Anat-64iso-S982_v2.npz")
ABIDE_data = arr['data']
ABIDE_subs = arr['subs']
nsubs = ABIDE_data.shape[0]
print([arr.shape for arr in [ABIDE_subs,ABIDE_data]])
print((ABIDE_data.min(),ABIDE_data.max()))

# Load Data Legend
df = pd.read_csv("/neurospin/psy_sbox/rl264746/ABIDE_legend_S982.csv", header=0)
df = df.iloc[np.array([df['BIDS_ID'].values[s] in ABIDE_subs for s in range(len(df))])]
df.reset_index(inplace=True)
print(df.shape)

assert len(df)==len(ABIDE_subs), 'lenght mismatch'
assert all([df['BIDS_ID'][s] == ABIDE_subs[s] for s in range(len(df))]), 'order mismatch'

patients = df['DxGroup'].values==1
controls = df['DxGroup'].values==2

TD_subs = ABIDE_data[controls,:,:,:] # Data of Typically Developing participants
DX_subs = ABIDE_data[patients,:,:,:] # Data of ASD participants

print(TD_subs.shape)
print(DX_subs.shape)

# TRAIN VAE
latent_dim = 32
batch_size = 16
disentangle = False
gamma = 100
encoder, decoder, vae = get_MRI_VAE_3D(input_shape=(64, 64, 64, 1),
                                       latent_dim=32,
                                       batch_size=batch_size,
                                       disentangle=True,
                                       gamma=gamma,
                                       kernel_size=3,
                                       filters=48,
                                       intermediate_dim=128,
                                       nlayers=2,
                                       bias=True)

loss = list()
fn = '../records/science_reproducibility/VAE_weights'

nbatches = 1e6
for i in tqdm(range(1, int(nbatches))):

    batch_idx = np.random.randint(low=0, high=ABIDE_data.shape[0], size=batch_size)
    data_batch = ABIDE_data[batch_idx, :, :, :]

    history = vae.train_on_batch(data_batch)
    mse = ((data_batch - vae.predict(data_batch)[:, :, :, :, 0]) ** 2).mean()

    if np.mod(i, 5) == 0:  # Plot training progress
        im1 = data_batch[0, 32, :, :]
        im = vae.predict(data_batch)[0, 32, :, :, 0]
        plot_trainProgress(loss, im, im1)

    if np.mod(i, 100) == 0:  # Save every 100 batches
        vae.save_weights(fn)

    if mse < .005:
        break


# train CVAE
latent_dim = 16
batch_size = 16
beta = 1
gamma = 100
disentangle = True
cvae, z_encoder, s_encoder, cvae_decoder = get_MRI_CVAE_3D(latent_dim=latent_dim, beta=beta,
                                                           disentangle=disentangle, gamma=gamma, bias=True,
                                                           batch_size=batch_size)
loss = list()
fn = '../records/science_reproducibility/CVAE_weights'

# initial check
DX_batch = DX_subs[np.random.randint(low=0,high=DX_subs.shape[0],size=batch_size),:,:,:]
TD_batch = TD_subs[np.random.randint(low=0,high=TD_subs.shape[0],size=batch_size),:,:,:]

if len(loss)==0:
    loss.append(np.nan)
    im,im1,ss = cvae_query(ABIDE_data,s_encoder,z_encoder,cvae_decoder)
    plot_trainProgress(loss,im,im1)
    loss = list()
else:
    im,im1,ss = cvae_query(ABIDE_data,s_encoder,z_encoder,cvae_decoder)
    plot_trainProgress(loss,im,im1)

nbatches = 1e6
for i in tqdm(range(1, int(nbatches))):

    DX_batch = DX_subs[np.random.randint(low=0, high=DX_subs.shape[0], size=batch_size), :, :, :]
    TD_batch = TD_subs[np.random.randint(low=0, high=TD_subs.shape[0], size=batch_size), :, :, :]

    hist = cvae.train_on_batch([DX_batch, TD_batch])
    loss.append(hist)

    mse = ((np.array([DX_batch, TD_batch]) - np.array(cvae.predict([DX_batch, TD_batch]))[:, :, :, :, :, 0]) ** 2).mean()

    assert not np.isnan(hist), 'loss is NaN - somethings wrong'

    im, im1, ss = cvae_query(ABIDE_data, s_encoder, z_encoder, cvae_decoder)

    if np.mod(i, 5) == 0:  # Plot training progress
        plot_trainProgress(loss, im, im1)
        plot_four(DX_batch, TD_batch, z_encoder, s_encoder, cvae_decoder, cvae, idx=0)
        plot_four(DX_batch, TD_batch, z_encoder, s_encoder, cvae_decoder, cvae, idx=1)

    if np.mod(i, 101) == 0:  # Save every 100 batches
        cvae.save_weights(fn)

    if mse < .005:
        break