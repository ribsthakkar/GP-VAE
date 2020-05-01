import os

import numpy as np

with np.load(os.path.join('hmnist', 'hmnist_rescale_tight.npz')) as data:
    np.savez(os.path.join('hmnist', 'hmnist_rescale_loose.npz'), **data)
