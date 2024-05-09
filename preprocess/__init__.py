import os

import numpy as np


def save_sparse(path, x):
    idx = np.where(x > 0)
    values = x[idx]
    np.savez(path, idx=idx, values=values, shape=x.shape)


def load_sparse(path):
    data = np.load(path, allow_pickle=True)
    idx, values = data['idx'], data['values']
    mat = np.zeros(data['shape'], dtype=values.dtype)
    mat[tuple(idx)] = values
    return mat


def save_data(path, code_x, visit_lens, y, hf_y, intervals):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'y'), y)
    np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    save_sparse(os.path.join(path, 'intervals'), intervals)
    # np.savez(os.path.join(path, 'code_mask'),code_mask=code_mask)
