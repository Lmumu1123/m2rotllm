import os
from dotenv import dotenv_values
from tqdm import tqdm
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil
import scipy.io as sio
import json


class PUCons:
    def __init__(self):
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_BEARING_DATASET_PATH"], "PU帕德博恩大学")
        file_info_file = os.path.join(self.root_dir, "fault.json")
        self.file_info = json.load(open(file_info_file, "r"))
        self.code = '6203'
        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []

    def process_all_data(self):
        for sub_dir in os.listdir(self.root_dir):
            if sub_dir == "fault.json":
                continue
            fault, severity = self.file_info[sub_dir]
            if fault * severity == 0:
                label = 0
            else:
                label = (fault - 1) * 3 + severity
            for file in tqdm(os.listdir(os.path.join(self.root_dir, sub_dir))):
                if file.endswith(".mat"):
                    file_infos = file.split("_")
                    rpm = int(file_infos[0].replace('N', '')) * 100 // 60
                    m = int(file_infos[1].replace('M', '')) / 10
                    f = int(file_infos[2].replace('F', '')) * 100
                    load = f'{m}Nm+{f}N'

                    meta_data_cache = f'{load}_{rpm}'

                    try:
                        private_condition_id = self.meta_data_cache_list.index(meta_data_cache)
                    except ValueError:
                        private_condition_id = len(self.meta_data_cache_list)
                        self.meta_data_cache_list.append(meta_data_cache)
                        self.meta_data_util.add_condition(private_condition_id, "PU", "Bearing", self.code, 0, rpm, load)

                    all_data = sio.loadmat(os.path.join(self.root_dir, sub_dir, file))
                    for key in all_data.keys():
                        if not key.endswith("__"):
                            data = all_data[key]["Y"][0][0][0][6][2][0]
                            data = data.flatten()
                            self.meta_data_util.add_data(private_condition_id, label, 64000, data)


if __name__ == '__main__':
    PUCons().process_all_data()
