import numpy as np
from dotenv import dotenv_values
import os
import sqlite3
from tqdm import tqdm


class MetaDataUtil:
    def __init__(self):
        config = dotenv_values()
        self.file_info_file = config["DATASET_FILE_INFO_PATH"]
        self.condition_file = config["DATASET_CONDITION_PATH"]
        self.raw_dataset_dir = config["RAW_DATASET_DIR"]
        self.check_path_exist()
        self.init_condition_id = self._init_condition_id()
        self.file_id = self._init_file_id()

        self.file_info_pointer = open(self.file_info_file, "a")
        self.condition_pointer = open(self.condition_file, "a")

    def check_path_exist(self):
        if not os.path.exists(self.file_info_file):
            f = open(self.file_info_file, "w")
            f.write("file,condition,label\n")
            f.close()
        if not os.path.exists(self.condition_file):
            f = open(self.condition_file, "w")
            f.write("condition,dataset,component,code,channel,rpm(Hz),load\n")
            f.close()
        os.makedirs(self.raw_dataset_dir, exist_ok=True)

    def _init_condition_id(self):
        f = open(self.condition_file, "r")
        lines = f.readlines()
        if len(lines) == 1:
            return 0
        else:
            max_id = 0
            for line in lines[1:]:
                condition_id = int(line.split(",")[0])
                if condition_id > max_id:
                    max_id = condition_id
            return max_id + 1

    def _init_file_id(self):
        f = open(self.file_info_file, "r")
        lines = f.readlines()
        if len(lines) == 1:
            return 0
        else:
            return int(lines[-1].split(",")[0]) + 1

    def add_condition(self, private_condition_id, dataset, component, code, channel, rpm, load):
        public_condition_id = self.init_condition_id + private_condition_id
        self.condition_pointer.write(
            f"{public_condition_id},{dataset},{component},{code},{channel},{rpm},{load}\n")

    def _add_file_info(self, private_condition_id, label):
        public_condition_id = self.init_condition_id + private_condition_id
        self.file_info_pointer.write(f"{self.file_id},{public_condition_id},{label}\n")
        self.file_id += 1

    def add_data(self, private_condition_id, label, sample_rate, data):
        data_length = len(data)
        data_num = data_length // sample_rate
        new_data = data[:data_num * sample_rate].reshape(data_num, sample_rate).astype(np.float32)
        for sample_data in new_data:
            save_path = os.path.join(self.raw_dataset_dir, f'{self.file_id:08d}.npy')
            np.save(save_path, sample_data)
            self._add_file_info(private_condition_id, label)

    def close_pointers(self):
        self.file_info_pointer.close()
        self.condition_pointer.close()


def meta_data_convert():
    config = dotenv_values()
    data_path = config["DATASET_METADATA_PATH"]
    open(data_path, "w").close()
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE condition (condition_id INTEGER PRIMARY KEY, dataset TEXT, component TEXT, "
                   "code TEXT, channel INTEGER, rpm TEXT, load TEXT)")
    cursor.execute("CREATE TABLE file_info (file_id INTEGER PRIMARY KEY, condition_id INTEGER, label INTEGER, "
                   "FOREIGN KEY (condition_id) REFERENCES condition(condition_id))")
    cursor.execute("CREATE TABLE label_note (label INTEGER PRIMARY KEY, note TEXT)")
    conn.commit()

    condition_info_path = config["DATASET_CONDITION_PATH"]
    condition_info_list = open(condition_info_path, 'r').readlines()
    for condition_info in tqdm(condition_info_list[1:]):
        condition = condition_info.strip().split(',')
        cursor.execute("INSERT INTO condition VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (condition[0], condition[1], condition[2], condition[3], condition[4], condition[5],
                        condition[6]))
    conn.commit()

    file_info_path = config["DATASET_FILE_INFO_PATH"]
    file_info_list = open(file_info_path, 'r').readlines()
    for file_info in tqdm(file_info_list[1:]):
        file = file_info.strip().split(',')
        cursor.execute("INSERT INTO file_info VALUES (?, ?, ?)", (file[0], file[1], file[2]))
    conn.commit()

    label_note_path = config["DATASET_DIR"] + "/label_note.txt"
    label_note_list = open(label_note_path, 'r').readlines()
    for label_note in tqdm(label_note_list):
        label, note = label_note.strip().split(" - ")
        cursor.execute("INSERT INTO label_note VALUES (?, ?)", (label, note))

    cursor.execute("CREATE INDEX condition_dataset_index ON condition (dataset)")
    cursor.execute("CREATE INDEX condition_component_index ON condition (component)")
    cursor.execute("CREATE INDEX file_condition_index ON file_info (condition_id)")
    cursor.execute("CREATE INDEX file_label_index ON file_info (label)")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    # meta_data_util = MetaDataUtil()
    meta_data_convert()
