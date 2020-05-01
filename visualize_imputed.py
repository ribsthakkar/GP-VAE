import os

import matplotlib.pyplot as plt
import numpy as np

sample = 3
frame = 4
experiment_imputed = 'models/200430_23_52_36_mixed/'
validation_type = 'both'
hmnistdir = "data/hmnist/hmnist_rescale.npz"
spdir = "data/sprites/sprites.npz"
fig, axes = plt.subplots(2, 2)
axes[1, 0].set_xlabel('image (\'full\') with mask applied')
axes[1, 1].set_xlabel('corrupted image (\'miss\')')
axes[0, 0].set_ylabel('SPRITES', rotation=90)
axes[1, 0].set_ylabel('HMNIST (rescaled)', rotation=90)
shape = (64, 64, 3)

if validation_type == 'both':
    h_data = np.load(hmnistdir)
    s_data = np.load(spdir)
    imputed_data = np.load(experiment_imputed + 'imputed.npy')
    x_val_full = np.concatenate((h_data['x_test_full'], s_data['x_test_full']))
    x_val_miss = np.concatenate((h_data['x_test_miss'], s_data['x_test_miss']))
    m_val_miss = np.concatenate((h_data['m_test_miss'], s_data['m_test_miss']))
    print(x_val_full.shape, x_val_miss.shape, m_val_miss.shape, imputed_data.shape)
    data = x_val_full[sample, frame].reshape(*shape)
    mask = x_val_miss[sample, frame].reshape(*shape)
    miss = m_val_miss[sample, frame].reshape(*shape)
    imp = imputed_data[sample, frame].reshape(*shape)
    axes[0, 0].imshow(data * (1 - mask))
    axes[0, 1].imshow(miss)
    axes[1, 0].imshow(imp)
    axes[1, 1].imshow(miss)

plt.suptitle(f'Sample: {sample}\nFrame: {frame}')
plt.show()
