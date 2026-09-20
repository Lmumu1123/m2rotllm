import threading
from dotenv import dotenv_values
import os
import numpy as np
from scipy.fft import dct
from tqdm import tqdm
import h5py


def get_h5_file_path():
    config = dotenv_values()
    dcn_path = config["DCN_DATASET_PATH"]
    return dcn_path


def get_file_list():
    config = dotenv_values()
    raw_dataset_dir = config["RAW_DATASET_DIR"]
    dcn_path = config["DCN_DATASET_PATH"]
    return raw_dataset_dir, os.listdir(raw_dataset_dir), dcn_path


def init_hd5(data_path, sample_num, target_rate):
    with h5py.File(data_path, 'w') as f:
        f.create_dataset('data', (sample_num, target_rate), dtype='float32')


def process_single_data(file_id, file_path, target_rate, h5_data):
    data = np.load(file_path)
    data = dct(data)
    if len(data) < target_rate:
        data = np.pad(data, (0, target_rate - len(data)))
    else:
        data = data[:target_rate]
    data_energy = np.sum(data ** 2)
    data = data * np.sqrt(target_rate / data_energy)
    data = data.astype(np.float32)
    h5_data[file_id] = data


def process_all_data(thread_num=16, target_rate=24000):
    data_dir, file_list, dcn_path = get_file_list()
    init_hd5(dcn_path, len(file_list), target_rate)
    h5_data = h5py.File(dcn_path, 'r+')['data']
    main_thread_list = []
    child_thread_list = []
    for file in file_list:
        child_thread_list.append(threading.Thread(target=process_single_data, args=(int(file.replace('.npy', '')),
                                                                                    os.path.join(data_dir, file),
                                                                                    target_rate,
                                                                                    h5_data)))
        if len(child_thread_list) == thread_num:
            main_thread_list.append(child_thread_list)
            child_thread_list = []
    main_thread_list.append(child_thread_list)

    for main_thread in tqdm(main_thread_list):
        for child_thread in main_thread:
            child_thread.start()
        for child_thread in main_thread:
            child_thread.join()


def random_plot():
    import matplotlib.pyplot as plt
    file_path = get_h5_file_path()
    with h5py.File(file_path, 'r') as f:
        data = f['data'][np.random.randint(0, len(f['data']))]
        plt.plot(data)
        plt.show()


def plot_hist():
    import matplotlib.pyplot as plt
    file_path = get_h5_file_path()
    f = h5py.File(file_path, 'r')
    sum_data = np.zeros(24000)
    for i in tqdm(range(len(f['data']))):
        if i % 100 == 0:
            data = f['data'][i]
            sum_data += np.abs(data)
    plt.plot(sum_data / (len(f['data']) // 100))
    f.close()
    plt.show()


def multipy_data():
    file_path = get_h5_file_path()
    f = h5py.File(file_path, 'r+')
    data = f['data']
    for i in tqdm(range(len(data))):
        data[i] = data[i] / 100
    f.close()


if __name__ == '__main__':
    # random_plot()
    # process_all_data()
    multipy_data()
    plot_hist()
