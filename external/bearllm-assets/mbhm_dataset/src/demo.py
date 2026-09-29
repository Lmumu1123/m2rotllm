"""
To run this script, you may need to edit the .env file to specify the path to the metadata and data files.
"""


import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import sqlite3
from dotenv import dotenv_values


def get_bearing_file_list():
    metadata_path = dotenv_values()['METADATA_PATH']
    conn = sqlite3.connect(metadata_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT f.file_id, f.label FROM file_info f "
        "JOIN condition c on f.condition_id = c.condition_id ")
    # "WHERE c.dataset != 'CWRU'") # If you want to filter out some dataset
    file_info = cursor.fetchall()
    conn.close()
    return file_info


def split_file_list(file_list):
    train_set = []
    val_set = []
    test_set = []
    for i in range(len(file_list)):
        if i % 10 < 7:
            train_set.append(file_list[i])
        elif i % 10 < 9:
            val_set.append(file_list[i])
        else:
            test_set.append(file_list[i])
    return train_set, val_set, test_set


class DemoDataset(Dataset):
    def __init__(self, subset_info):
        self.info = subset_info
        data_path = dotenv_values()['DATA_PATH']
        self.data = h5py.File(data_path, 'r')['vibration']

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        file_info = self.info[idx]
        data = self.data[file_info[0]]
        data = torch.from_numpy(data).to(torch.float32).reshape(1, -1)
        label = file_info[1]
        return data, label


if __name__ == '__main__':
    train_set, val_set, test_set = split_file_list(get_bearing_file_list())
    print(f"Train set: {len(train_set)}")
    print(f"Validation set: {len(val_set)}")
    print(f"Test set: {len(test_set)}")

    test_loader = DataLoader(DemoDataset(test_set), batch_size=32, shuffle=True)
    for data, label in test_loader:
        print(data.shape, label)
        break
