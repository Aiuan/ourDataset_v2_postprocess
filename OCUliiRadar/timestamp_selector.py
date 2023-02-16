import os
import numpy as np

class DataFolder(object):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = os.listdir(self.folder_path)
        self.files.remove('logs.csv')
        self.files.sort()
        self.ts_str = np.char.replace(np.array(self.files), '.pcd', '')
        self.ts = self.ts_str.astype('float64')

    def select_by_ts(self, ts_str):
        tmp = float(ts_str) - self.ts
        idx = np.argmin(np.abs(tmp))
        error = tmp[idx]
        # assert np.abs(error) < 0.05
        path_matched = os.path.join(self.folder_path, self.files[idx])
        ts_str_matched = self.ts_str[idx]
        return path_matched, error, ts_str_matched

def test():
    OCULiiRadar_root = 'F:\\20221217\\OCULiiRadar_pcd'
    ts_str = '1671267290.996'

    data_folder = DataFolder(OCULiiRadar_root)
    path_matched, error, ts_str_matched = data_folder.select_by_ts(ts_str)

    print('done')

if __name__ == '__main__':
    test()
