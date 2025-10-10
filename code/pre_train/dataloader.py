import sqlite3
import h5pickle
from dotenv import dotenv_values
import numpy as np
from torch.utils.data import Dataset
import torch

default_filter_str = (
    "SELECT f.file_id, f.label FROM file_info f "
    "JOIN condition c on f.condition_id = c.condition_id ")


def get_filtered_sample_list(filter_str=''):
    filter_str = default_filter_str + filter_str
    metadata_path = dotenv_values()['DATASET_METADATA_PATH']
    conn = sqlite3.connect(metadata_path)
    cursor = conn.cursor()
    cursor.execute(filter_str)
    file_info = cursor.fetchall()
    conn.close()
    return np.array(file_info)


def split_file_list(file_list):
    batch_num = file_list.shape[0] // 10
    f = file_list[:batch_num * 10]
    f = f.reshape(batch_num, 10, 2)
    train_samples = f[:, :7].reshape(-1, 2)
    val_samples = f[:, 7:9].reshape(-1, 2)
    test_samples = f[:, 9].reshape(-1, 2)
    return train_samples, val_samples, test_samples


class VibrationDataset(Dataset):
    def __init__(self, subset_info, domain='freq', snr=None):
        self.info = subset_info
        data_path = dotenv_values()['DCN_DATASET_PATH']
        if domain == 'time':
            data_path = dotenv_values()['TIME_DATASET_PATH']
        elif snr is not None:
            data_path = data_path.replace('data.hdf5', f'data_noise_{snr}.hdf5')
        self.data = h5pickle.File(data_path, 'r')['data']

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        file_info = self.info[idx]
        data = self.data[file_info[0]]
        data = torch.from_numpy(data).to(torch.float32).unsqueeze(0)
        label = file_info[1]
        return data, torch.tensor(label, dtype=torch.long)
