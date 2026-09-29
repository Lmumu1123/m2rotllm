import os
import numpy as np
from dotenv import dotenv_values
from tqdm import tqdm
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil


def get_label(filename):
    if '-B' in filename:
        return 5
    if '-O' in filename:
        return 8
    if '-I' in filename:
        return 2
    if '-N' in filename:
        return 0
    raise ValueError(f"Unknown label for {filename}")


def get_info(filename):
    infos = filename.split('-')
    rpm = int(infos[2].replace('rpm', ''))
    load = infos[3]
    return rpm, load


class JUSTCons:
    def __init__(self):
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_BEARING_DATASET_PATH"], "JUST江苏科技大学转轴轴承")
        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []

    def process_all_data(self):
        for condition_id in range(6):
            condition_dir = os.path.join(self.root_dir, f'工况{condition_id + 1}')
            for filename in tqdm(os.listdir(condition_dir)):
                if filename.endswith(".csv"):
                    label = get_label(filename)
                    rpm, load = get_info(filename)
                    file_path = os.path.join(condition_dir, filename)
                    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
                    data = np.float32(data)
                    total_time = float(data[-1, 0])
                    total_samples = len(data)
                    sample_rate = round(total_samples / total_time)
                    for channel in range(6):
                        channel_data = data[:, channel + 1]

                        meta_data_cache = f'{condition_id}_{channel}'

                        try:
                            private_condition_id = self.meta_data_cache_list.index(meta_data_cache)
                        except ValueError:
                            self.meta_data_cache_list.append(meta_data_cache)
                            private_condition_id = len(self.meta_data_cache_list) - 1
                            self.meta_data_util.add_condition(private_condition_id, 'JUST', 'Bearing', '-', channel, rpm, load)

                        self.meta_data_util.add_data(private_condition_id, label, sample_rate, channel_data)


if __name__ == '__main__':
    JUSTCons().process_all_data()