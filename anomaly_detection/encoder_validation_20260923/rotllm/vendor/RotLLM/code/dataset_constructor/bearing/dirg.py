import os
import numpy as np
from dotenv import dotenv_values
import scipy.io as sio
from tqdm import tqdm
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil

dirg_freq = 51200

dirg_fault_dict = {
    "C0A": 0,
    "C1A": 3,
    "C2A": 2,
    "C3A": 1,
    "C4A": 6,
    "C5A": 5,
    "C6A": 4,
}


def calc_load(voltage_str):
    volt = int(voltage_str)
    rpm_hat = round(volt * 2 / 100)
    return str(rpm_hat * 100) + "N"


class DIRGCons:
    def __init__(self):
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_BEARING_DATASET_PATH"], "DIRG", "Total Data", "RAW")

        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []

    def process_single_file(self, file_name):
        rpm = int(file_name.split("_")[1])
        load = calc_load(file_name.split("_")[2])

        file_type = file_name.split("\\")[-1].split("_")[0]
        label = dirg_fault_dict[file_type]

        all_data = sio.loadmat(os.path.join(self.root_dir, file_name))

        for key in all_data.keys():
            if "__" not in key:
                data = np.array(all_data[key], dtype=np.float32)
                chs = data.shape[1]
                for ch_id in range(chs):

                    meta_data_cache = f'{load}_{rpm}_{ch_id}'
                    if meta_data_cache not in self.meta_data_cache_list:
                        self.meta_data_cache_list.append(meta_data_cache)
                        private_condition_id = len(self.meta_data_cache_list) - 1
                        self.meta_data_util.add_condition(private_condition_id, "DIRG", "Bearing", '-', ch_id, rpm,
                                                          load)
                    else:
                        private_condition_id = self.meta_data_cache_list.index(meta_data_cache)

                    ch_data = data[:, ch_id]
                    self.meta_data_util.add_data(private_condition_id, label, dirg_freq, ch_data)

    def process_all_data(self):
        for file in tqdm(os.listdir(self.root_dir)):
            if file.endswith(".mat"):
                self.process_single_file(file)
        self.meta_data_util.close_pointers()


if __name__ == '__main__':
    DIRGCons().process_all_data()
