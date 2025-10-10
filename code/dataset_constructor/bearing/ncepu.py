import os
import numpy as np
from dotenv import dotenv_values
from tqdm import tqdm
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil

code = "NTN NU204 ET2X"
sub_dirs = ["Ball", "Inner", "Outer", "Normal"]
freq = 100000


def decode_filename(file_name):
    if "N" in file_name:
        fault = 0
    elif "OB" in file_name:
        fault = 8
    elif "IB" in file_name:
        fault = 2
    else:
        fault = 5
    if "1500" in file_name:
        rpm = 1500 // 60
    elif "1000" in file_name:
        rpm = 1000 // 60
    else:
        rpm = 500 // 60
    if 'D' in file_name:
        load = "0-3000N"
    elif "F" in file_name:
        load = "1500N"
    else:
        load = "0"
    return rpm, fault, load


class NCEPUCons:
    def __init__(self):
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_BEARING_DATASET_PATH"], "JNU江南大学数据集", "Total data")
        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []

    def process_all_data(self):
        for sub_dir in sub_dirs:
            file_list = os.listdir(os.path.join(self.root_dir, sub_dir))
            for file_name in tqdm(file_list):
                rpm, label, load = decode_filename(file_name)
                file_path = os.path.join(self.root_dir, sub_dir, file_name)
                with open(file_path, "r") as f:
                    raw_data = f.readlines()
                    file_data = []
                    for line in raw_data:
                        line = line.strip()
                        line = line.strip(",")
                        line = line.split(",")
                        if len(line) != 5:
                            continue
                        line = [float(x) for x in line]
                        file_data.append(line)
                    file_data = np.array(file_data, dtype=np.float32)
                    chs = file_data.shape[1]
                    for ch_id in range(chs):

                        meta_data_cache = f'{ch_id}_{rpm}_{load}'
                        try:
                            private_condition_id = self.meta_data_cache_list.index(meta_data_cache)
                        except ValueError:
                            self.meta_data_cache_list.append(meta_data_cache)
                            private_condition_id = len(self.meta_data_cache_list) - 1
                            self.meta_data_util.add_condition(private_condition_id, "NCEPU", "Bearing", code, ch_id,
                                                              rpm,
                                                              load)
                        ch_data = file_data[:, ch_id]
                        self.meta_data_util.add_data(private_condition_id, label, freq, ch_data)


if __name__ == '__main__':
    ncepu = NCEPUCons()
    ncepu.process_all_data()
