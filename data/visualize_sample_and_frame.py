import os

import matplotlib.pyplot as plt
import numpy as np

# pick the sample (0-8999) and frame (0-7) to plot
sample = 2000
frame = 4

fig, axes = plt.subplots(2, 2)
axes[1, 0].set_xlabel('image (\'full\') with mask applied')
axes[1, 1].set_xlabel('corrupted image (\'miss\')')
axes[0, 0].set_ylabel('SPRITES', rotation=90)
axes[1, 0].set_ylabel('HMNIST (rescaled)', rotation=90)
shape = (64, 64, 3)

with np.load(os.path.join('sprites', 'sprites.npz')) as stor:
    data = stor['x_train_full'][sample, frame].reshape(*shape)
    mask = stor['m_train_miss'][sample, frame].reshape(*shape)
    miss = stor['x_train_miss'][sample, frame].reshape(*shape)
    axes[0, 0].imshow(data * (1 - mask))
    axes[0, 1].imshow(miss)

with np.load(os.path.join('hmnist', 'hmnist_rescale_loose.npz')) as stor:
    data = stor['x_train_full'][sample, frame].reshape(*shape)
    mask = stor['m_train_miss'][sample, frame].reshape(*shape)
    miss = stor['x_train_miss'][sample, frame].reshape(*shape)
    axes[1, 0].imshow(data * (1 - mask))
    axes[1, 1].imshow(miss)

plt.suptitle(f'Sample: {sample}\nFrame: {frame}')
plt.show()
