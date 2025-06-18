import torch
from scipy import io as sio
import numpy as np


def load_data(data_name):
    if data_name == 'Urban':
        file = './data/Urban.mat'

    dataset = sio.loadmat(file)
    data, GT, abundance = dataset['img_3d'], dataset['endmember'], dataset['abundance']
    init_em = dataset['init_em']
    init_em = torch.from_numpy(init_em).unsqueeze(2).unsqueeze(3).float()

    data = data.transpose([1, 2, 0])
    n_rows, n_cols, n_bands = data.shape
    abundance = np.reshape(abundance, [abundance.shape[0], n_rows, n_cols])
    GT = GT.transpose([1, 0])

    return data, GT, abundance, init_em
