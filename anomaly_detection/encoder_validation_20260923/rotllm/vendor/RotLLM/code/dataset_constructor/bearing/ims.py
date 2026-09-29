import os
import numpy as np
from dotenv import dotenv_values
from tqdm import tqdm
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil

sub_dirs = ["1st_test", "2nd_test", "3rd_test"]

freq = 20000
rpm = 2000 // 60
load = "6000lbs"
code = "Rexnord ZA-2115"


def get_fault_label(t_name, ch_id, t_id):
    if t_name == "1st_test":
        if ch_id in [0, 1, 2, 3]:
            return 0
        elif ch_id in [4, 5]:
            if t_id < 1750:
                return 0
            else:
                return 1
        else:
            if t_id < 1500:
                return 0
            elif t_id < 2000:
                return 5
            else:
                return 6
    elif t_name == "2nd_test":
        if ch_id == 0:
            if t_id < 500:
                return 0
            elif t_id < 700:
                return 7
            elif t_id < 900:
                return 8
            else:
                return 9
        else:
            return 0
    else:
        if t_id < 6150:
            return 0
        else:
            if ch_id == 2:
                if t_id < 6250:
                    return 8
                else:
                    return 9
            else:
                if t_id < 6275:
                    return 7
                else:
                    return 8


class IMSCons:
    def __init__(self):
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_BEARING_DATASET_PATH"], "IMS", "Total Data")

        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []

    def process_all_data(self):
        for i in range(3):
            sub_dir = sub_dirs[i]
            file_list = os.listdir(os.path.join(self.root_dir, sub_dir))
            file_list.sort()
            for t_id in tqdm(range(len(file_list))):
                file_name = file_list[t_id]
                file_path = os.path.join(self.root_dir, sub_dir, file_name)
                file_data = np.loadtxt(file_path, dtype=np.float32)
                chs = file_data.shape[1]
                for ch_id in range(chs):
                    ch_data = file_data[:, ch_id]

                    meta_data_cache = f'{sub_dir}_{ch_id}'
                    try:
                        private_condition_id = self.meta_data_cache_list.index(meta_data_cache)
                    except ValueError:
                        self.meta_data_cache_list.append(meta_data_cache)
                        private_condition_id = len(self.meta_data_cache_list) - 1
                        self.meta_data_util.add_condition(private_condition_id, "IMS", "Bearing", code, ch_id, rpm,
                                                          load)

                    label = get_fault_label(sub_dir, ch_id, t_id)
                    self.meta_data_util.add_data(private_condition_id, label, freq, ch_data)


if __name__ == '__main__':
    IMSCons().process_all_data()
