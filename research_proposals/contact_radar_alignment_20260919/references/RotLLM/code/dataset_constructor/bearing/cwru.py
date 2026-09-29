import json
import numpy as np
from dotenv import dotenv_values
import os
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil
import scipy.io as sio
import tqdm


def get_cwru_rpm(data: dict):
    for key in data.keys():
        if key.endswith("RPM"):
            return int(data[key][0, 0]) // 60
    return 1700//60


class CWRUCons:
    def __init__(self):
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_BEARING_DATASET_PATH"], "CWRU", "raw")
        file_info_json = os.path.join(self.root_dir, "CWRU.json")
        self.all_file_info = json.load(open(file_info_json, "r", encoding="utf-8"))

        self.freq_list = [48000, 12000, 48000, 12000]
        self.code_list = ['-', '6205-2RS JEM SKF', '6205-2RS JEM SKF', '6203-2RS JEM SKF']
        self.load_list = self.all_file_info['_work_load']

        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []

    def get_file_info(self, file_name: str):
        file_id: str = file_name.split(".")[0]

        file_info: list = self.all_file_info[str(file_id)]
        freq = self.freq_list[file_info[0]]
        code = self.code_list[file_info[0]]

        work_load = self.load_list[file_info[1]]

        fault_id: int = file_info[2]
        severity_id: int = file_info[3]

        if severity_id * fault_id == 0:
            label = 0
        else:
            if severity_id == 4:
                severity_id = 3
            label = (fault_id - 1) * 3 + severity_id
        return freq, label, work_load, code

    def process_single_file(self, file_name):
        freq, label, work_load, code = self.get_file_info(file_name)
        all_data = sio.loadmat(os.path.join(self.root_dir, 'data', file_name))
        rpm = get_cwru_rpm(all_data)
        ch_id = 0
        for key in all_data.keys():
            if key.endswith("_time"):
                meta_data_cache = f'{work_load}_{rpm}_{code}_{ch_id}'
                if meta_data_cache not in self.meta_data_cache_list:
                    self.meta_data_cache_list.append(meta_data_cache)
                    private_condition_id = self.meta_data_cache_list.index(meta_data_cache)
                    self.meta_data_util.add_condition(private_condition_id, 'CWRU', 'Bearing', code, ch_id, rpm,
                                                      work_load)
                else:
                    private_condition_id = self.meta_data_cache_list.index(meta_data_cache)

                data = all_data[key].reshape(-1)
                data = np.array(data)
                self.meta_data_util.add_data(private_condition_id, label, freq, data)
                ch_id += 1

    def process_all_data(self):
        for file_name in tqdm.tqdm(os.listdir(os.path.join(self.root_dir, 'data'))):
            if file_name.endswith(".mat"):
                self.process_single_file(file_name)


if __name__ == "__main__":
    CWRUCons().process_all_data()

