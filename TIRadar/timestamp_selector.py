import os
import numpy as np

class DataFolder(object):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.subfolders = os.listdir(self.folder_path)
        self.subfolders.sort()

        self.files_path = None
        self.ts_str = None
        self.group = None
        for subfolder in self.subfolders:
            files = os.listdir(os.path.join(self.folder_path, subfolder))
            files.sort()
            files = np.array(files)
            files_path = np.char.add(os.path.join(self.folder_path, subfolder), np.char.add(os.path.sep, files))
            group = np.array([subfolder for i in range(len(files))])

            if self.ts_str is None:
                self.ts_str = np.char.replace(files, '.adcdata.npz', '')
            else:
                self.ts_str = np.concatenate((self.ts_str, np.char.replace(files, '.adcdata.npz', '')))

            if self.files_path is None:
                self.files_path = files_path
            else:
                self.files_path = np.concatenate((self.files_path, files_path))

            if self.group is None:
                self.group = group
            else:
                self.group = np.concatenate((self.group, group))

        self.ts = self.ts_str.astype('float64')

    def select_by_ts(self, ts_str):
        tmp = float(ts_str) - self.ts
        idx = np.argmin(np.abs(tmp))
        error = tmp[idx]
        # assert np.abs(error) < 0.05
        path_matched = self.files_path[idx]
        ts_str_matched = self.ts_str[idx]
        return path_matched, error, ts_str_matched


def test():
    TIRadar_root = 'F:\\20221217\\TIRadar_npz'
    ts_str = '1671267290.996'

    data_folder = DataFolder(TIRadar_root)
    path_matched, error, ts_str_matched = data_folder.select_by_ts(ts_str)

    print('done')

if __name__ == '__main__':
    test()
