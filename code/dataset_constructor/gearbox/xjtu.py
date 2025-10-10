from dotenv import dotenv_values
import os

from torch.utils.hipify.hipify_python import meta_data

from utils.data.dataset_constructor.meta_data_util import MetaDataUtil
import numpy as np
import tqdm


def get_file_info(file_path):
    if "brokentooth" in file_path:
        label = 13
    elif "missingtooth" in file_path:
        label = 14
    elif "normalstate" in file_path:
        label = 0
    elif "rootcracks" in file_path:
        label = 11
    elif "toothwear" in file_path:
        label = 12
    else:
        raise ValueError(f"Unknown label for {file_path}")

    if "Chan1" in file_path:
        channel = 0
    elif "Chan2" in file_path:
        channel = 1
    else:
        raise ValueError(f"Unknown channel for {file_path}")
    return label, channel


class XJTUCons:
    def __init__(self):
        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []
        self.sample_rate = 20480
        self.rpm = 1800 // 30
        self.load = "3HP"
        self.code = "DDS"
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_GEARBOX_DATASET_PATH"], "XJTU Gearbox", "XJTU_Gearbox")

    def process_single_data(self, file_path):
        label, channel = get_file_info(file_path)
        data = np.loadtxt(file_path, skiprows=14, )
        meta_data_cache = f'{self.load}_{self.rpm}_{channel}_{self.code}'
        try:
            meta_data_cache_id = self.meta_data_cache_list.index(meta_data_cache)
        except ValueError:
            self.meta_data_cache_list.append(meta_data_cache)
            meta_data_cache_id = len(self.meta_data_cache_list) - 1
            self.meta_data_util.add_condition(meta_data_cache_id, "XJTU", "Gear", self.code, channel, self.rpm, self.load)

        self.meta_data_util.add_data(meta_data_cache_id, label, self.sample_rate, data)


    def process_all_data(self):
        for subdir in tqdm.tqdm(os.listdir(self.root_dir)):
            if "Bearing" in subdir:
                continue
            for file in os.listdir(os.path.join(self.root_dir, subdir)):
                if file.endswith(".txt"):
                    self.process_single_data(os.path.join(self.root_dir, subdir, file))


if __name__ == "__main__":
    XJTUCons().process_all_data()
