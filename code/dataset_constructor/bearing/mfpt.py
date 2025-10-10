import os
from dotenv import dotenv_values
from tqdm import tqdm
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil
import scipy.io as sio

fault_dict = {
    "1 - Three Baseline Conditions": 0,
    "2 - Three Outer Race Fault Conditions": 8,
    "3 - Seven More Outer Race Fault Conditions": 8,
    "4 - Seven Inner Race Fault Conditions": 2,
}

code = 'NK12/12'


class MFPTCons:
    def __init__(self):
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_BEARING_DATASET_PATH"], "MFPT")
        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []

    def process_all_data(self):
        for sub_dir in fault_dict.keys():
            file_list = os.listdir(os.path.join(self.root_dir, sub_dir))
            for file in tqdm(file_list):
                if file.endswith(".mat"):
                    all_data = sio.loadmat(os.path.join(self.root_dir, sub_dir, file))
                    fault = fault_dict[sub_dir]
                    freq = int(all_data["bearing"]["sr"][0][0][0][0])
                    data = all_data["bearing"]["gs"][0][0].flatten()
                    rpm = int(all_data["bearing"]["rate"][0][0][0][0])
                    load = str(all_data["bearing"]["load"][0][0][0]).replace('[', '').replace(']', '')+'lbs'

                    meta_data_cache = f'{load}_{rpm}'
                    try:
                        private_condition_id = self.meta_data_cache_list.index(meta_data_cache)
                    except ValueError:
                        private_condition_id = len(self.meta_data_cache_list)
                        self.meta_data_cache_list.append(meta_data_cache)
                        self.meta_data_util.add_condition(private_condition_id, "MFPT", "Bearing", code, 0, rpm, load)

                    self.meta_data_util.add_data(private_condition_id, fault, freq, data)


if __name__ == '__main__':
    MFPTCons().process_all_data()
