import os
import json
import time
import glob

import numpy as np
import cv2
import scipy.io as scio

def log(text):
    print(text)

def log_BLUE(text):
    print('\033[0;34;40m{}\033[0m'.format(text))

def log_YELLOW(text):
    print('\033[0;33;40m{}\033[0m'.format(text))

def log_GREEN(text):
    print('\033[0;32;40m{}\033[0m'.format(text))

def log_RED(text):
    print('\033[0;31;40m{}\033[0m'.format(text))

class GroupFolder(object):
    def __init__(self, path_group_folder):
        self.path_group_folder = path_group_folder
        self.names_frame_folder = os.listdir(self.path_group_folder)
        self.names_frame_folder.sort()
        self.num_frames = len(self.names_frame_folder)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        path_frame_folder = os.path.join(self.path_group_folder, self.names_frame_folder[idx])
        return FrameFolder(path_frame_folder)

class FrameFolder(object):
    def __init__(self, path_frame_folder):
        self.path_frame_folder = path_frame_folder
        self.names_sensor_folder = os.listdir(self.path_frame_folder)
        self.names_sensor_folder.sort()

    def read_sensor(self, name_sensors=None):
        if name_sensors is None:
            name_sensors = self.names_sensor_folder
        elif isinstance(name_sensors, str):
            name_sensors = [name_sensors]

        assert isinstance(name_sensors, list)
        res = dict()
        for name_sensor in name_sensors:
            path_sensor_folder = os.path.join(self.path_frame_folder, name_sensor)
            filenames = os.listdir(path_sensor_folder)
            res[name_sensor] = [os.path.join(path_sensor_folder, filename) for filename in filenames]

        return res


def load_IRayCamera_png(path):
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return data

def load_LeopardCamera_png(path):
    data = cv2.imread(path)
    # BGR -> RGB
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return data

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def load_VelodyneLidar_pcd(path):
    with open(path, "r") as f:
        data = f.readlines()
    data = data[10:]
    data = np.array(data)
    data = np.char.replace(data, '\n', '')
    data = np.char.split(data, ' ')
    data = np.array(data.tolist())

    x = np.array(data[:, 0], dtype=np.float32)
    y = np.array(data[:, 1], dtype=np.float32)
    z = np.array(data[:, 2], dtype=np.float32)
    intensity = np.array(data[:, 3], dtype=np.uint8)
    idx_laser = np.array(data[:, 4], dtype=np.uint8)
    unix_timestamp = np.array(data[:, 5], dtype=np.float64)

    res = {
        'x': x,
        'y': y,
        'z': z,
        'intensity': intensity,
        'idx_laser': idx_laser,
        'unix_timestamp': unix_timestamp
    }
    return res

def load_OCULiiRadar_pcd(path):
    with open(path, "r") as f:
        data = f.readlines()
    data = data[10:]
    data = np.array(data)
    data = np.char.replace(data, '\n', '')
    data = np.char.split(data, ' ')
    data = np.array(data.tolist())

    x = np.array(data[:, 0], dtype=np.float32)
    y = np.array(data[:, 1], dtype=np.float32)
    z = np.array(data[:, 2], dtype=np.float32)
    doppler = np.array(data[:, 3], dtype=np.float32)
    snr = np.array(data[:, 4], dtype=np.float32)

    res = {
        'x': x,
        'y': y,
        'z': z,
        'doppler': doppler,
        'snr': snr
    }
    return res

def load_TIRadar_adcdata(path):
    data = np.load(path, allow_pickle=True)

    res = {
        'data_imag': data['data_imag'],
        'data_real': data['data_real'],
        'mode_infos': data['mode_infos'][()]
    }
    return res

def load_TIRadar_calibmat(path):
    data = scio.loadmat(path)

    calibResult = dict()
    names = data['calibResult'].dtype.names
    values = data['calibResult'].tolist()[0][0]
    for i in range(len(values)):
        calibResult[names[i]] = values[i]
    calibResult['RangeMat'] = calibResult['RangeMat'].astype(np.int)

    params = dict()
    names = data['params'].dtype.names
    values = data['params'].tolist()[0][0]
    for i in range(len(values)):
        params[names[i]] = values[i][0, 0]

    res = {
        'calibResult': calibResult,
        'params': params
    }

    return res

def load_TIRadar_heatmapBEV(path):
    data = np.load(path, allow_pickle=True)

    res = {
        'x': data['x'],
        'y': data['y'],
        'heatmapBEV_static': data['heatmapBEV_static'],
        'heatmapBEV_dynamic': data['heatmapBEV_dynamic']
    }
    return res

def load_TIRadar_pcd(path):
    with open(path, "r") as f:
        data = f.readlines()
    data = data[10:]
    data = np.array(data)
    data = np.char.replace(data, '\n', '')
    data = np.char.split(data, ' ')
    data = np.array(data.tolist())

    x = np.array(data[:, 0], dtype=np.float32)
    y = np.array(data[:, 1], dtype=np.float32)
    z = np.array(data[:, 2], dtype=np.float32)
    doppler = np.array(data[:, 3], dtype=np.float32)
    snr = np.array(data[:, 4], dtype=np.float32)
    intensity = np.array(data[:, 5], dtype=np.float32)
    noise = np.array(data[:, 6], dtype=np.float32)

    res = {
        'x': x,
        'y': y,
        'z': z,
        'doppler': doppler,
        'snr': snr,
        'intensity': intensity,
        'noise': noise
    }
    return res

def unix2local(unix_ts_str):
    # unix_timestamp_str --> local_timestamp_str
    return '{}.{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(unix_ts_str))), unix_ts_str.split('.')[-1])

def pcd_in_zone(pcd_dict, xlim=[-np.inf, np.inf], ylim=[-np.inf, np.inf], zlim=[-np.inf, np.inf]):
    assert 'x' in pcd_dict.keys()
    assert 'y' in pcd_dict.keys()
    assert 'z' in pcd_dict.keys()

    mask_x = np.logical_and(
        pcd_dict['x'] >= xlim[0],
        pcd_dict['x'] <= xlim[1]
    )

    mask_y = np.logical_and(
        pcd_dict['y'] >= ylim[0],
        pcd_dict['y'] <= ylim[1]
    )

    mask_z = np.logical_and(
        pcd_dict['z'] >= zlim[0],
        pcd_dict['z'] <= zlim[1]
    )

    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    res = np.array(list(pcd_dict.values())).T
    res = res[mask, :]

    return res

def undistort_image(image, intrinsic, radial_distortion, tangential_distortion):
    k1, k2, k3 = radial_distortion
    p1, p2 = tangential_distortion

    image_undistort = cv2.undistort(image, intrinsic, np.array([k1, k2, p1, p2, k3]))

    return image_undistort

def save_dict_as_json(json_path, dict_data):
    data = json.dumps(dict_data, sort_keys=True, indent=4)
    with open(json_path, 'w', newline='\n') as f:
        f.write(data)


if __name__ == '__main__':
    import glob
    import matplotlib.pyplot as plt

    folder = 'F:\\dataset_v2\\20221217_group0000_mode1_280frames\\frame0000\\LeopardCamera1'
    json_path = glob.glob(os.path.join(folder, '*.json'))[0]
    image_path = glob.glob(os.path.join(folder, '*.png'))[0]
    data_json = load_json(json_path)
    LeopardCamera0_image = load_LeopardCamera_png(image_path)

    image_undistort = undistort_image(
        LeopardCamera0_image,
        np.array(data_json['intrinsic_matrix']),
        np.array(data_json['radial_distortion']),
        np.array(data_json['tangential_distortion'])
    )

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(LeopardCamera0_image)

    plt.subplot(2, 1, 2)
    plt.imshow(image_undistort)


    plt.show()


class Group(object):
    def __init__(self, root):
        self.root = root
        self.frame_foldernames = os.listdir(self.root)
        self.num_frames = len(self.frame_foldernames)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx_frame):
        return Frame(os.path.join(self.root, 'frame{:>04d}'.format(idx_frame)))

class Frame(object):
    def __init__(self, root):
        self.root = root

        self.IRayCamera_png_path = self.__find_and_check_path__(os.path.join(self.root, 'IRayCamera', '*.png'))
        self.IRayCamera_json_path = self.__find_and_check_path__(os.path.join(self.root, 'IRayCamera', '*.json'))

        self.LeopardCamera0_png_path = self.__find_and_check_path__(os.path.join(self.root, 'LeopardCamera0', '*.png'))
        self.LeopardCamera0_json_path = self.__find_and_check_path__(os.path.join(self.root, 'LeopardCamera0', '*.json'))

        self.LeopardCamera1_png_path = self.__find_and_check_path__(os.path.join(self.root, 'LeopardCamera1', '*.png'))
        self.LeopardCamera1_json_path = self.__find_and_check_path__(os.path.join(self.root, 'LeopardCamera1', '*.json'))

        self.MEMS_json_path = self.__find_and_check_path__(os.path.join(self.root, 'MEMS', '*.json'))

        self.OCULiiRadar_pcd_path = self.__find_and_check_path__(os.path.join(self.root, 'OCULiiRadar', '*.pcd'))
        self.OCULiiRadar_json_path = self.__find_and_check_path__(os.path.join(self.root, 'OCULiiRadar', '*.json'))

        self.TIRadar_adcdata_path = self.__find_and_check_path__(os.path.join(self.root, 'TIRadar', '*.adcdata.npz'))
        self.TIRadar_pcd_path = self.__find_and_check_path__(os.path.join(self.root, 'TIRadar', '*.pcd'))
        self.TIRadar_heatmapBEV_path = self.__find_and_check_path__(os.path.join(self.root, 'TIRadar', '*.heatmapBEV.npz'))
        self.TIRadar_json_path = self.__find_and_check_path__(os.path.join(self.root, 'TIRadar', '*.json'))
        self.TIRadar_calibmat_path = self.__find_and_check_path__(os.path.join(self.root, 'TIRadar', '*.mat'))
        
        self.VelodyneLidar_pcd_path = self.__find_and_check_path__(os.path.join(self.root, 'VelodyneLidar', '*.pcd'))
        self.VelodyneLidar_json_path = self.__find_and_check_path__(os.path.join(self.root, 'VelodyneLidar', '*.json'))

    def __find_and_check_path__(self, path):
        res = glob.glob(path)
        if len(res) == 1:
            return res[0]
        else:
            log_YELLOW('{} find {} results'.format(path, len(res)))
            return None

    def get_sensor_data(self, sensor_data_name):
        if sensor_data_name == 'IRayCamera_png' and self.IRayCamera_png_path is not None:
            return load_IRayCamera_png(self.IRayCamera_png_path)

        if sensor_data_name == 'LeopardCamera0_png' and self.LeopardCamera0_png_path is not None:
            return load_LeopardCamera_png(self.LeopardCamera0_png_path)

        if sensor_data_name == 'LeopardCamera1_png' and self.LeopardCamera1_png_path is not None:
            return load_LeopardCamera_png(self.LeopardCamera1_png_path)

        if sensor_data_name == 'OCULiiRadar_pcd' and self.OCULiiRadar_pcd_path is not None:
            return load_OCULiiRadar_pcd(self.OCULiiRadar_pcd_path)

        if sensor_data_name == 'TIRadar_adcdata' and self.TIRadar_adcdata_path is not None:
            return load_TIRadar_adcdata(self.TIRadar_adcdata_path)

        if sensor_data_name == 'TIRadar_pcd' and self.TIRadar_pcd_path is not None:
            return load_TIRadar_pcd(self.TIRadar_pcd_path)

        if sensor_data_name == 'TIRadar_heatmapBEV' and self.TIRadar_heatmapBEV_path is not None:
            return load_TIRadar_heatmapBEV(self.TIRadar_heatmapBEV_path)

        if sensor_data_name == 'TIRadar_calibmat' and self.TIRadar_calibmat_path is not None:
            return load_TIRadar_calibmat(self.TIRadar_calibmat_path)

        if sensor_data_name == 'VelodyneLidar_pcd' and self.VelodyneLidar_pcd_path is not None:
            return load_VelodyneLidar_pcd(self.VelodyneLidar_pcd_path)

        if sensor_data_name.split('_')[1] == 'json' and self.__getattribute__('{}_path'.format(sensor_data_name)) is not None:
            return load_json(self.__getattribute__('{}_path'.format(sensor_data_name)))

        return None



        