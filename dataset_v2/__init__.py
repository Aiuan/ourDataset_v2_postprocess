import os
import json
import time
import glob

import numpy as np
import cv2
import pandas as pd
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

def load_pcd(path, pcd_sensor):
    if pcd_sensor == 'VelodyneLidar':
        return load_VelodyneLidar_pcd(path)
    elif pcd_sensor == 'OCULiiRadar':
        return load_OCULiiRadar_pcd(path)
    elif pcd_sensor == 'TIRadar':
        return load_TIRadar_pcd(path)
    else:
        log_RED("pcd_sensor not in ['VelodyneLidar', 'OCULiiRadar', 'TIRadar']")
        return None


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
    return '{}.{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(unix_ts_str))),
                          unix_ts_str.split('.')[-1])

def pcd_in_zone(pcd_dict, xlim=[-np.inf, np.inf], ylim=[-np.inf, np.inf], zlim=[-np.inf, np.inf], return_type='np_array'):
    assert 'x' in pcd_dict.keys()
    assert 'y' in pcd_dict.keys()
    assert 'z' in pcd_dict.keys()
    assert return_type == 'np_array' or return_type == 'dict'

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

    if return_type == 'np_array':
        res = np.array(list(pcd_dict.values())).T
        res = res[mask, :]
    else:
        res = dict()
        for key, value in pcd_dict.items():
            res[key] = pcd_dict[key][mask]

    return res

def rectangular2polar(x, y, z):
    '''
        x: right, y: front, z: up
    '''
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))

    # elevation (-90, 90)
    # elev = 0, y+
    # elev = -90, z-
    # elev = 90, z+
    sin_elev = z / r
    elev = np.arcsin(sin_elev) / np.pi * 180

    # azimuth (-180, 180]
    # azim = 0, y +
    # azim = 90, x +
    # azim = -90, x -
    # azim = 180, y -
    sin_azim = x / r / np.cos(elev / 180 * np.pi)
    cos_azim = y / r / np.cos(elev / 180 * np.pi)
    azim = np.arcsin(sin_azim) / np.pi * 180
    mask = (cos_azim < 0)
    azim[mask] = azim[mask] / np.abs(azim[mask]) * (180 - np.abs(azim[mask]))

    return azim, elev, r

def pcd_in_polar_zone(pcd_dict, azimlim=[-np.inf, np.inf], elevlim=[-np.inf, np.inf], rlim=[-np.inf, np.inf], return_type='np_array', add_aer=False):
    assert 'x' in pcd_dict.keys()
    assert 'y' in pcd_dict.keys()
    assert 'z' in pcd_dict.keys()
    assert return_type == 'np_array' or return_type == 'dict'

    azim, elev, r = rectangular2polar(pcd_dict['x'], pcd_dict['y'], pcd_dict['z'])
    if add_aer:
        pcd_dict['azimuth'] = azim
        pcd_dict['elevation'] = elev
        pcd_dict['range'] = r

    mask_azim = np.logical_and(azim >= azimlim[0], azim <= azimlim[1])

    mask_elev = np.logical_and(elev >= elevlim[0], elev <= elevlim[1])

    mask_r = np.logical_and(r >= rlim[0], r <= rlim[1])

    mask = np.logical_and(np.logical_and(mask_azim, mask_elev), mask_r)

    if return_type == 'np_array':
        res = np.array(list(pcd_dict.values())).T
        res = res[mask, :]
    else:
        res = dict()
        for key, value in pcd_dict.items():
            res[key] = pcd_dict[key][mask]

    return res


def undistort_image(image, intrinsic, radial_distortion, tangential_distortion):
    k1, k2, k3 = radial_distortion
    p1, p2 = tangential_distortion

    image_undistort = cv2.undistort(image, np.array(intrinsic), np.array([k1, k2, p1, p2, k3]))

    return image_undistort


def save_dict_as_json(json_path, dict_data):
    data = json.dumps(dict_data, sort_keys=True, indent=4)
    with open(json_path, 'w', newline='\n') as f:
        f.write(data)


class Group(object):
    def __init__(self, root):
        self.root = root
        self.frame_foldernames = os.listdir(self.root)
        self.frame_foldernames.sort()
        self.num_frames = len(self.frame_foldernames)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx_frame):
        return Frame(os.path.join(self.root, 'frame{:>04d}'.format(idx_frame)))

    def get_route(self):
        route = {
            'timestamp': [],
            'latitude_N': [],
            'longitude_E': [],
            'height': [],
            'north_vel': [],
            'east_vel': [],
            'up_vel': [],
            'pitch': [],
            'roll': [],
            'azimuth': []
        }
        for idx_frame in range(self.__len__()):
            frame = self.__getitem__(idx_frame)
            MEMS_json = frame.get_sensor_data('MEMS_json')
            route['timestamp'].append(MEMS_json['timestamp'])
            route['latitude_N'].append(MEMS_json['msg_ins']['latitude_N'])
            route['longitude_E'].append(MEMS_json['msg_ins']['longitude_E'])
            route['height'].append(MEMS_json['msg_ins']['height'])
            route['north_vel'].append(MEMS_json['msg_ins']['north_vel'])
            route['east_vel'].append(MEMS_json['msg_ins']['east_vel'])
            route['up_vel'].append(MEMS_json['msg_ins']['up_vel'])
            route['pitch'].append(MEMS_json['msg_ins']['pitch'])
            route['roll'].append(MEMS_json['msg_ins']['roll'])
            route['azimuth'].append(MEMS_json['msg_ins']['azimuth'])

        route = pd.DataFrame(route)
        return route


class Frame(object):
    def __init__(self, root):
        self.root = root

        self.IRayCamera_png_path = self.__find_and_check_path__(os.path.join(self.root, 'IRayCamera', '*.png'))
        self.IRayCamera_json_path = self.__find_and_check_path__(os.path.join(self.root, 'IRayCamera', '*.json'))

        self.LeopardCamera0_png_path = self.__find_and_check_path__(os.path.join(self.root, 'LeopardCamera0', '*.png'))
        self.LeopardCamera0_json_path = self.__find_and_check_path__(
            os.path.join(self.root, 'LeopardCamera0', '*.json'))

        self.LeopardCamera1_png_path = self.__find_and_check_path__(os.path.join(self.root, 'LeopardCamera1', '*.png'))
        self.LeopardCamera1_json_path = self.__find_and_check_path__(
            os.path.join(self.root, 'LeopardCamera1', '*.json'))

        self.MEMS_json_path = self.__find_and_check_path__(os.path.join(self.root, 'MEMS', '*.json'))

        self.OCULiiRadar_pcd_path = self.__find_and_check_path__(os.path.join(self.root, 'OCULiiRadar', '*.pcd'))
        self.OCULiiRadar_json_path = self.__find_and_check_path__(os.path.join(self.root, 'OCULiiRadar', '*.json'))

        self.TIRadar_adcdata_path = self.__find_and_check_path__(os.path.join(self.root, 'TIRadar', '*.adcdata.npz'))
        self.TIRadar_pcd_path = self.__find_and_check_path__(os.path.join(self.root, 'TIRadar', '*.pcd'))
        self.TIRadar_heatmapBEV_path = self.__find_and_check_path__(
            os.path.join(self.root, 'TIRadar', '*.heatmapBEV.npz'))
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

        if sensor_data_name.split('_')[1] == 'json' and self.__getattribute__(
                '{}_path'.format(sensor_data_name)) is not None:
            return load_json(self.__getattribute__('{}_path'.format(sensor_data_name)))

        return None


def pcd_transform(pcd_dict, extrinsic):
    assert 'x' in pcd_dict.keys()
    assert 'y' in pcd_dict.keys()
    assert 'z' in pcd_dict.keys()

    xyz1 = np.stack((
        pcd_dict['x'],
        pcd_dict['y'],
        pcd_dict['z'],
        np.ones((pcd_dict['x'].shape[0]))
    ))

    xyz1_new = np.matmul(extrinsic, xyz1)

    pcd_dict['x'] = xyz1_new[0, :]
    pcd_dict['y'] = xyz1_new[1, :]
    pcd_dict['z'] = xyz1_new[2, :]

    return pcd_dict


def pcd_projection(x, y, z, intrinsic, extrinsic):
    n = x.shape[0]

    xyz1 = np.stack((x, y, z, np.ones((n))))

    projection_matrix = np.matmul(intrinsic, extrinsic[0:3, :])

    UVZ = np.matmul(projection_matrix, xyz1)
    uv1 = UVZ[0:2, :] / UVZ[2, :]

    u = uv1[0, :].round()
    v = uv1[1, :].round()

    return u, v


def pcd_in_image(pcd_dict, image_width, image_height, intrinsic, extrinsic, add_uv=False):
    assert 'x' in pcd_dict.keys()
    assert 'y' in pcd_dict.keys()
    assert 'z' in pcd_dict.keys()

    u, v = pcd_projection(pcd_dict['x'], pcd_dict['y'], pcd_dict['z'], intrinsic, extrinsic)
    if add_uv:
        pcd_dict['u'] = u
        pcd_dict['v'] = v

    mask_u = np.logical_and(
        pcd_dict['u'] >= 0,
        pcd_dict['u'] <= image_width
    )

    mask_v = np.logical_and(
        pcd_dict['v'] >= 0,
        pcd_dict['v'] <= image_height
    )

    mask = np.logical_and(mask_u, mask_v)

    for key, value in pcd_dict.items():
        pcd_dict[key] = value[mask]

    return pcd_dict


def save_dict_as_VelodyneLidar_pcd(pcd_path, pcd_dict):
    pcd = pd.DataFrame({
        'x': pcd_dict['x'].astype('float32'),
        'y': pcd_dict['y'].astype('float32'),
        'z': pcd_dict['z'].astype('float32'),
        'intensity': pcd_dict['intensity'].astype('uint8'),
        'idx_laser': pcd_dict['idx_laser'].astype('uint8'),
        'unix_timestamp': pcd_dict['unix_timestamp'].astype('float64'),
    })

    pcd.to_csv(pcd_path, sep=' ', index=False, header=False)
    with open(pcd_path, 'r') as f_pcd:
        lines = f_pcd.readlines()

    with open(pcd_path, 'w') as f_pcd:
        f_pcd.write('VERSION .7\n')
        f_pcd.write('FIELDS')
        for col in pcd.columns.values:
            f_pcd.write(' {}'.format(col))
        f_pcd.write('\n')
        f_pcd.write('SIZE 4 4 4 1 1 8\n')
        f_pcd.write('TYPE F F F U U F\n')
        f_pcd.write('COUNT 1 1 1 1 1 1\n')
        f_pcd.write('WIDTH {}\n'.format(len(pcd)))
        f_pcd.write('HEIGHT 1\n')
        f_pcd.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f_pcd.write('POINTS {}\n'.format(len(pcd)))
        f_pcd.write('DATA ascii\n')
        f_pcd.writelines(lines)
