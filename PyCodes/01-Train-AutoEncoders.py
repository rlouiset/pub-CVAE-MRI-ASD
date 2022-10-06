# GPU checks
import tensorflow as tf

tf.test.gpu_device_name()

# GPU checks
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from make_models2 import *

# Make tqdm work for notebooks
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

import numpy as np
import pandas as pd

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


def cpu():
    with tf.device('/cpu:0'):
        random_image_cpu = tf.random.normal((100, 100, 100, 3))
        net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
        return tf.math.reduce_sum(net_cpu)


def gpu():
    with tf.device('/device:GPU:0'):
        random_image_gpu = tf.random.normal((100, 100, 100, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)


# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time / gpu_time)))

# LOAD Data
arr = np.load("/neurospin/psy_sbox/rl264746/ABIDE-Anat-64iso-S982.npz")
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
fn = '../recorde/science_reproducibility/VAE_weights'

nbatches = 1e6
for i in tqdm(range(1, nbatches)):

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