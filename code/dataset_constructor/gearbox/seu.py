from dotenv import dotenv_values
import os
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil
import numpy as np
import tqdm


def get_file_info(file_path):
    if "Chipped" in file_path:
        label = 13
    elif "Health" in file_path:
        label = 0
    elif "Miss" in file_path:
        label = 14
    elif "Root" in file_path:
        label = 11
    elif "Surface" in file_path:
        label = 10
    else:
        raise ValueError(f"Unknown label for {file_path}")

    if "20_0" in file_path:
        rpm = 20
        load = 0
    elif "30_2" in file_path:
        rpm = 30
        load = "7.32Nm"
    else:
        raise ValueError(f"Unknown rpm and load for {file_path}")

    return rpm, load, label


class SEUCons:
    def __init__(self):
        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []
        self.sample_rate = 5120
        self.code = "DDS"
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_GEARBOX_DATASET_PATH"], "SEU东南大学", "gearset")

    def process_single_data(self, file_path):
        rpm, load, label = get_file_info(file_path)
        data = np.loadtxt(file_path, delimiter='\t', skiprows=16, usecols=(1, 2, 3)).T
        for channel in range(3):
            channel_data = data[channel]

            meta_data_cache = f'{load}_{rpm}_{channel}'
            try:
                meta_data_cache_id = self.meta_data_cache_list.index(meta_data_cache)
            except ValueError:
                self.meta_data_cache_list.append(meta_data_cache)
                meta_data_cache_id = len(self.meta_data_cache_list) - 1
                self.meta_data_util.add_condition(meta_data_cache_id, "SEU", "Gear", self.code, channel, rpm, load)

            self.meta_data_util.add_data(meta_data_cache_id, label, self.sample_rate, channel_data)

    def process_all_data(self):
        for file in tqdm.tqdm(os.listdir(self.root_dir)):
            if file.endswith(".csv"):
                file_path = os.path.join(self.root_dir, file)
                self.process_single_data(file_path)


if __name__ == "__main__":
    SEUCons().process_all_data()
