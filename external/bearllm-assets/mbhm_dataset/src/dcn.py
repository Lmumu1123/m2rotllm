import numpy as np
from scipy.fft import dct


def dcn(raw_data, target_rate=24000):
    data_std = np.std(raw_data)
    data_mean = np.mean(raw_data)
    data = (raw_data - data_mean) / data_std
    data = dct(data)
    if len(data) < target_rate:
        data = np.pad(data, (0, target_rate - len(data)))
    else:
        data = data[:target_rate]
    data_energy = np.sum(data ** 2)
    data = data * np.sqrt(target_rate / data_energy) / 100
    data = data.astype(np.float32)
    return data


if __name__ == '__main__':
    rand_data = np.random.rand(123456)
    print(dcn(rand_data).shape)