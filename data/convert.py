import os
import time

import numpy as np


HMNIST_PATH = os.path.join('hmnist', 'hmnist_mnar.npz')
ZEROS = np.zeros(shape=(64, 64, 3))
N1 = 9000 # reduce 60000 to 9000
N2 = 2664 # reduce 10000 to 2664
F = 8

def _retrieve_by_key(path, key):
    with np.load(path) as data:
        output = data[key]
    return output

def _get_train_indices():
    y = _retrieve_by_key(HMNIST_PATH, 'y_train')
    output = [] 
    for i in range(10): 
        i_indices, = np.where(y == i) 
        output.extend(np.random.choice(i_indices, (N1 + i) // 10, replace=False)) 
    return output

def _get_test_indices():
    y = _retrieve_by_key(HMNIST_PATH, 'y_test')
    output = []
    for i in range(10):
        i_indices, = np.where(y == i)
        output.extend(np.random.choice(i_indices, (N2 + i) // 10, replace=False))
    return output

TRAIN_INDICES = _get_train_indices()
TEST_INDICES = _get_test_indices()

def get_x_train_full():
    x_train_full = _retrieve_by_key(HMNIST_PATH, 'x_train_full')
    new_x_train_full = np.zeros(shape=(N1, 8, 64 * 64 * 3))
    for i, j in enumerate(TRAIN_INDICES):
        if (i + 1) % 100 == 0 or (i + 1) == N1:
            print(i + 1)
        for f in range(F):
            tmp = x_train_full[j, f]
            new = ZEROS.copy()
            new[18:-18, 18:-18, :] = np.repeat(tmp.reshape(28, 28)[:, :, np.newaxis], 3, axis=2)
            new_x_train_full[i, f] = new.reshape(-1)
    return new_x_train_full.astype('float32')

def get_x_train_miss():
    x_train_miss = _retrieve_by_key(HMNIST_PATH, 'x_train_miss')
    new_x_train_miss = np.zeros(shape=(N1, 8, 64 * 64 * 3))
    for i, j in enumerate(TRAIN_INDICES):
        if (i + 1) % 100 == 0 or (i + 1) == N1:
            print(i + 1)
        for f in range(F):
            tmp = x_train_miss[j, f]
            new = ZEROS.copy()
            new[18:-18, 18:-18, :] = np.repeat(tmp.reshape(28, 28)[:, :, np.newaxis], 3, axis=2)
            new_x_train_miss[i, f] = new.reshape(-1)
    return new_x_train_miss.astype('float32')

def get_m_train_miss():
    m_train_miss = _retrieve_by_key(HMNIST_PATH, 'm_train_miss')
    new_m_train_miss = np.zeros(shape=(N1, 8, 64 * 64 * 3))
    for i, j in enumerate(TRAIN_INDICES):
        if (i + 1) % 100 == 0 or (i + 1) == N1:
            print(i + 1)
        for f in range(F):
            tmp = m_train_miss[j, f]
            new = ZEROS.copy()
            new[18:-18, 18:-18, :] = np.repeat(tmp.reshape(28, 28)[:, :, np.newaxis], 3, axis=2)
            new_m_train_miss[i, f] = new.reshape(-1)
    return new_m_train_miss.astype('float32')

def get_y_train():
    y_train = _retrieve_by_key(HMNIST_PATH, 'y_train')
    return y_train[TRAIN_INDICES].astype('uint8')

def get_x_test_full():
    x_test_full = _retrieve_by_key(HMNIST_PATH, 'x_test_full')
    new_x_test_full = np.zeros(shape=(N2, 8, 64 * 64 * 3))
    for i, j in enumerate(TEST_INDICES):
        if (i + 1) % 100 == 0 or (i + 1) == N2:
            print(i + 1)
        for f in range(F):
            tmp = x_test_full[j, f]
            new = ZEROS.copy()
            new[18:-18, 18:-18, :] = np.repeat(tmp.reshape(28, 28)[:, :, np.newaxis], 3, axis=2)
            new_x_test_full[i, f] = new.reshape(-1)
    return new_x_test_full.astype('float32')

def get_x_test_miss():
    x_test_miss = _retrieve_by_key(HMNIST_PATH, 'x_test_miss')
    new_x_test_miss = np.zeros(shape=(N2, 8, 64 * 64 * 3))
    for i, j in enumerate(TEST_INDICES):
        if (i + 1) % 100 == 0 or (i + 1) == N2:
            print(i + 1)
        for f in range(F):
            tmp = x_test_miss[j, f]
            new = ZEROS.copy()
            new[18:-18, 18:-18, :] = np.repeat(tmp.reshape(28, 28)[:, :, np.newaxis], 3, axis=2)
            new_x_test_miss[i, f] = new.reshape(-1)
    return new_x_test_miss.astype('float32')

def get_m_test_miss():
    m_test_miss = _retrieve_by_key(HMNIST_PATH, 'm_test_miss')
    new_m_test_miss = np.zeros(shape=(N2, 8, 64 * 64 * 3))
    for i, j in enumerate(TEST_INDICES):
        if (i + 1) % 100 == 0 or (i + 1) == N2:
            print(i + 1)
        for f in range(F):
            tmp = m_test_miss[j, f]
            new = ZEROS.copy()
            new[18:-18, 18:-18, :] = np.repeat(tmp.reshape(28, 28)[:, :, np.newaxis], 3, axis=2)
            new_m_test_miss[i, f] = new.reshape(-1)
    return new_m_test_miss.astype('float32')

def get_y_test():
    y_test = _retrieve_by_key(HMNIST_PATH, 'y_test')
    return y_test[TEST_INDICES].astype('uint8')


t0 = time.time()
print('Time:', 0)
np.savez_compressed(
    os.path.join('hmnist', 'hmnist_rescale.npz'),
    x_train_full=get_x_train_full(),
    x_train_miss=get_x_train_miss(),
    m_train_miss=get_m_train_miss(),
    y_train=get_y_train(),
    x_test_full=get_x_test_full(),
    x_test_miss=get_x_test_miss(),
    m_test_miss=get_m_test_miss(),
    y_test=get_y_test(),
)
print('Time:', time.time() - t0)
