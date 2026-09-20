import os
from dotenv import dotenv_values
from tqdm import tqdm
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil
import numpy as np

freq = 25600

rpm_dict = {
    "35Hz12kN": (35, "12000N"),
    "37.5Hz11kN": (37.5, "11000N"),
    "40Hz10kN": (40, "10000N"),
}

fault_dict = {
    "Bearing1_1": 3,
    "Bearing1_2": 3,
    "Bearing1_3": 3,
    "Bearing1_4": 2,
    "Bearing1_5": [1, 3],
    "Bearing2_1": 1,
    "Bearing2_2": 3,
    "Bearing2_3": 2,
    "Bearing2_4": 3,
    "Bearing2_5": 3,
    "Bearing3_1": 3,
    "Bearing3_2": [1, 2, 3],
    "Bearing3_3": 1,
    "Bearing3_4": 1,
    "Bearing3_5": 3,
}


class RefStd:
    def __init__(self, ref_dir):
        ref_data = np.loadtxt(os.path.join(ref_dir, "1.csv"), delimiter=",", skiprows=1)
        self.std = np.std(ref_data)

    def get_severity(self, data):
        data_std = np.std(data)
        severity = 0
        if data_std > 4 * self.std:
            severity = 3
        elif data_std > 2.5 * self.std:
            severity = 2
        elif data_std > 1.5 * self.std:
            severity = 1
        return severity


class XJTUCons:
    def __init__(self):
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_BEARING_DATASET_PATH"], "XJTU-SY西交昇阳", "Total Data")
        self.code = "LDK UER204"

        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []

    def process_all_data(self):
        for freq_dir, info in rpm_dict.items():
            rpm, load = info

            sub_dir = os.path.join(self.root_dir, freq_dir)
            for test_dir in os.listdir(sub_dir):
                fault = fault_dict[test_dir]
                if isinstance(fault, list):
                    continue
                data_dir = os.path.join(sub_dir, test_dir)
                ref = RefStd(data_dir)
                for file in tqdm(os.listdir(data_dir)):
                    if file.endswith(".csv"):
                        data = np.loadtxt(os.path.join(data_dir, file), delimiter=",", skiprows=1)
                        severity = ref.get_severity(data)
                        if severity == 0:
                            label = 0
                        else:
                            label = (fault - 1) * 3 + severity
                        for channel in range(2):
                            meta_data_cache = f"{load}_{rpm}_{channel}"
                            try:
                                private_condition_id = self.meta_data_cache_list.index(meta_data_cache)
                            except ValueError:
                                private_condition_id = len(self.meta_data_cache_list)
                                self.meta_data_cache_list.append(meta_data_cache)
                                self.meta_data_util.add_condition(private_condition_id, "XJTU", "Bearing", self.code,
                                                                  channel, rpm, load)
                            ch_data = data[:, channel]
                            self.meta_data_util.add_data(private_condition_id, label, freq, ch_data)


if __name__ == '__main__':
    XJTUCons().process_all_data()
