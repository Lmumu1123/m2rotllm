from dotenv import dotenv_values
import os
from utils.data.dataset_constructor.meta_data_util import MetaDataUtil
import scipy.io as sio
import tqdm


def get_file_info(file_path):
    file_name = os.path.basename(file_path)
    file_name = file_name.replace(".mat", "")
    file_infos = file_name.split(" ")
    if "normal" in file_infos[0]:
        label = 0
    else:
        if "pitting" in file_infos[0]:
            label = 10
        elif "fracture" in file_infos[0]:
            label = 13
        else:
            label = 12
    rpm = file_infos[1]
    rpm = f'{(int(rpm)+20)//60}'
    load = file_infos[2]
    return rpm, load, label


class SDUSTCons:
    def __init__(self):
        self.meta_data_util = MetaDataUtil()
        self.meta_data_cache_list = []
        self.codes = ["2-20-18 Sun", "2-20-17 plane tray"]
        self.sample_rate = 25600

        config = dotenv_values()
        self.root_dir = os.path.join(config["PUBLIC_GEARBOX_DATASET_PATH"], "SDUST山东科技大学")

        self.file_list = []
        self.init_file_list()

    def init_file_list(self):
        for sub_dir in os.listdir(self.root_dir):
            for file in os.listdir(os.path.join(self.root_dir, sub_dir)):
                if file.endswith(".mat") and "~" not in file and "-" not in file and "flu" not in file:
                    self.file_list.append((sub_dir, file))

    def process_single_data(self, file_path):
        rpm, load, label = get_file_info(file_path)
        if "太" in file_path:
            code = self.codes[0]
        elif "NC" in file_path:
            code = "-"
        else:
            code = self.codes[1]
        all_data = sio.loadmat(file_path)['vibration'].T
        channel_num = all_data.shape[0]
        for channel_id in range(channel_num):
            data = all_data[channel_id]
            meta_data_cache = f'{load}_{rpm}_{channel_id}_{code}'
            try:
                private_condition_id = self.meta_data_cache_list.index(meta_data_cache)
            except ValueError:
                private_condition_id = len(self.meta_data_cache_list)
                self.meta_data_cache_list.append(meta_data_cache)
                self.meta_data_util.add_condition(private_condition_id, "SDUST", "Gear", code, channel_id, rpm, load)

            self.meta_data_util.add_data(private_condition_id, label, self.sample_rate, data)

    def process_all_data(self):
        for sub_dir, file in tqdm.tqdm(self.file_list):
            file_path = os.path.join(self.root_dir, sub_dir, file)
            self.process_single_data(file_path)


if __name__ == '__main__':
    SDUSTCons().process_all_data()