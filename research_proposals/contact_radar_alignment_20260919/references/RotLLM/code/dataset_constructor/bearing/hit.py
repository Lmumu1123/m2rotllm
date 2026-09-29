import os
import numpy as np
from dotenv import dotenv_values
from tqdm import tqdm
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil

hit_freq = 20000
labels = [0, 2, 8]

bearing_code = 'inter-shaft 6306'


class HITCons:
    def __init__(self):
        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_BEARING_DATASET_PATH"], "HIT哈工大航发轴间轴承")

        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []

    def process_single_file(self, file_name):
        all_data = np.load(os.path.join(self.root_dir, file_name))

        for sample in all_data:
            l_rpm = sample[6][0]
            h_rpm = sample[6][1]
            rpm = int(h_rpm - l_rpm) // 200 * 200 // 60
            fault = int(sample[7][0])
            label = labels[fault]

            for channel in range(4):

                meta_data_cache = f'{rpm}_{channel}'

                try:
                    private_condition_id = self.meta_data_cache_list.index(meta_data_cache)
                except ValueError:
                    self.meta_data_cache_list.append(meta_data_cache)
                    private_condition_id = len(self.meta_data_cache_list) - 1
                    self.meta_data_util.add_condition(private_condition_id, "HIT", "Bearing", bearing_code, channel, rpm, '-')

                channel_data = sample[channel + 2]
                self.meta_data_util.add_data(private_condition_id, label, hit_freq, channel_data)

    def process_all_data(self):
        file_list = os.listdir(self.root_dir)
        for file in tqdm(file_list):
            if file.endswith(".npy"):
                self.process_single_file(file)


if __name__ == '__main__':
    HITCons().process_all_data()
