import os
import shutil
import json

import numpy as np
import pandas as pd

from Camera.timestamp_selector import DataFolder as CDF
from MEMS.timestamp_selector import DataFolder as MDF
from OCUliiRadar.timestamp_selector import DataFolder as ODF
from TIRadar.timestamp_selector import DataFolder as TDF
from VelodyneLidar.timestamp_selector import DataFolder as VDF
from utils import *

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_dict_as_json(json_path, dict_data):
    with open(json_path, 'w') as f:
        json.dump(dict_data, f)

def select_camera_by_TIRadar(root_Camera, root_TIRadar):
    tdf = TDF(root_TIRadar)
    cdf = CDF(root_Camera)

    path_matched_list = []
    error_list = []
    ts_Camera_list = []
    for ts_str_TIRadar in tdf.ts_str:
        path_matched, error, ts_str_Camera = cdf.select_by_ts(ts_str_TIRadar)
        path_matched_list.append(path_matched)
        error_list.append(error)
        ts_Camera_list.append(float(ts_str_Camera))

    return np.array(path_matched_list), np.array(error_list), np.array(ts_Camera_list)

def select_MEMS_by_TIRadar(root_MEMS, root_TIRadar):
    tdf = TDF(root_TIRadar)
    mdf = MDF(root_MEMS)

    path_matched_list = []
    error_list = []
    ts_MEMS_list = []
    for ts_str_TIRadar in tdf.ts_str:
        path_matched, error, ts_str_MEMS = mdf.select_by_ts(ts_str_TIRadar)
        path_matched_list.append(path_matched)
        error_list.append(error)
        ts_MEMS_list.append(float(ts_str_MEMS))

    return np.array(path_matched_list), np.array(error_list), np.array(ts_MEMS_list)

def select_OCULiiRadar_by_TIRadar(root_OCULiiRadar, root_TIRadar):
    tdf = TDF(root_TIRadar)
    odf = ODF(root_OCULiiRadar)

    path_matched_list = []
    error_list = []
    ts_OCULiiRadar_list = []
    for ts_str_TIRadar in tdf.ts_str:
        path_matched, error, ts_str_OCULii = odf.select_by_ts(ts_str_TIRadar)
        path_matched_list.append(path_matched)
        error_list.append(error)
        ts_OCULiiRadar_list.append(float(ts_str_OCULii))

    return np.array(path_matched_list), np.array(error_list), np.array(ts_OCULiiRadar_list)

def select_VelodyneLidar_by_TIRadar(root_VelodyneLidar, root_TIRadar):
    tdf = TDF(root_TIRadar)
    vdf = VDF(root_VelodyneLidar)

    path_matched_list = []
    error_list = []
    ts_VelodyneLidar_list = []
    for ts_str_TIRadar in tdf.ts_str:
        path_matched, error, ts_str_VelodyneLidar = vdf.select_by_ts(ts_str_TIRadar)
        path_matched_list.append(path_matched)
        error_list.append(error)
        ts_VelodyneLidar_list.append(float(ts_str_VelodyneLidar))

    return np.array(path_matched_list), np.array(error_list), np.array(ts_VelodyneLidar_list)

def main():
    root_TIRadar = 'D:\TIRadar_npz'
    root_IRayCamera = 'F:\\20221217\\IRayCamera'
    root_LeopardCamera0 = 'F:\\20221217\\LeopardCamera0'
    root_LeopardCamera1 = 'F:\\20221217\\LeopardCamera1'
    root_OCULiiRadar = 'F:\\20221217\\OCULiiRadar_pcd'
    root_MEMS = 'F:\\20221217\\MEMS_json'
    root_VelodyneLidar = 'F:\\20221217\\VelodyneLidar_pcd'

    root_calibration = 'F:\\sensors_calibration_v2\\results\\zhoushan_20221217_20221221'

    root_output = 'F:\\dataset_v2'
    if not os.path.exists(root_output):
        os.makedirs(root_output)

    tdf = TDF(root_TIRadar)

    IRayCamera_files_path, IRayCamera_error, IRayCamera_ts = select_camera_by_TIRadar(root_IRayCamera, root_TIRadar)
    LeopardCamera0_files_path, LeopardCamera0_error, LeopardCamera0_ts = select_camera_by_TIRadar(root_LeopardCamera0, root_TIRadar)
    LeopardCamera1_files_path, LeopardCamera1_error, LeopardCamera1_ts = select_camera_by_TIRadar(root_LeopardCamera1, root_TIRadar)

    MEMS_files_path, MEMS_error, MEMS_ts = select_MEMS_by_TIRadar(root_MEMS, root_TIRadar)

    OCULiiRadar_files_path, OCULiiRadar_error, OCULiiRadar_ts = select_OCULiiRadar_by_TIRadar(root_OCULiiRadar, root_TIRadar)

    VelodyneLidar_files_path, VelodyneLidar_error, VelodyneLidar_ts = select_VelodyneLidar_by_TIRadar(root_VelodyneLidar, root_TIRadar)

    df = pd.DataFrame(
        {
            'TIRadar_group': tdf.group,
            'TIRadar_ts': tdf.ts,
            'TIRadar_files_path': tdf.files_path,

            'IRayCamera_ts': IRayCamera_ts,
            'IRayCamera_files_path': IRayCamera_files_path,
            'IRayCamera_error': IRayCamera_error,

            'LeopardCamera0_ts': LeopardCamera0_ts,
            'LeopardCamera0_files_path': LeopardCamera0_files_path,
            'LeopardCamera0_error': LeopardCamera0_error,

            'LeopardCamera1_ts': LeopardCamera1_ts,
            'LeopardCamera1_files_path': LeopardCamera1_files_path,
            'LeopardCamera1_error': LeopardCamera1_error,

            'MEMS_ts': MEMS_ts,
            'MEMS_files_path': MEMS_files_path,
            'MEMS_error': MEMS_error,

            'OCULiiRadar_ts': OCULiiRadar_ts,
            'OCULiiRadar_files_path': OCULiiRadar_files_path,
            'OCULiiRadar_error': OCULiiRadar_error,

            'VelodyneLidar_ts': VelodyneLidar_ts,
            'VelodyneLidar_files_path': VelodyneLidar_files_path,
            'VelodyneLidar_error': VelodyneLidar_error,
        }
    )

    error_thred = 0.05
    mask = np.logical_or(
        np.logical_or(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        df['IRayCamera_error'].values > error_thred,
                        df['LeopardCamera0_error'].values > error_thred,
                    ),
                    df['LeopardCamera1_error'].values > error_thred,
                ),
                df['MEMS_error'].values > error_thred,
            ),
            df['OCULiiRadar_error'].values > error_thred,
        ),
        df['VelodyneLidar_error'].values > error_thred,
    )
    df_illegal = df[mask]

    df_legal = df[np.logical_not(mask)]

    # grouping
    v_loc = df_legal.index.values
    cut_points = np.where((v_loc[1:] - v_loc[:-1]) > 1)[0]
    continue_group = np.zeros((len(df_legal, )), dtype='int32')
    for cut_point in cut_points:
        continue_group[cut_point+1:] += 1
    df_legal.insert(loc=0, column='continue_group', value=continue_group)

    cnt_group_frames = []
    info_group = dict()
    for id_group, (group_name, group_df) in enumerate(df_legal.groupby(['continue_group', 'TIRadar_group'])):
        cnt_group_frames.append(len(group_df))
        info_group['group_{:>04d}'.format(id_group)] = group_df

        # create dataset_v2
        group_folder_name = '{}_group{:>04d}_{}_{}frames'.format(
            group_name[1].split('_')[0],
            id_group,
            group_name[1].split('_')[2],
            len(group_df)
        )
        group_folder_path = os.path.join(root_output, group_folder_name)
        if not os.path.exists(group_folder_path):
            os.makedirs(group_folder_path)

        log_BLUE('='*100)
        log_BLUE('{}'.format(group_folder_name))

        for id_frame in range(len(group_df)):
            group_frame_folder_name = 'frame{:>04d}'.format(id_frame)
            group_frame_folder_path = os.path.join(group_folder_path, group_frame_folder_name)
            if not os.path.exists(group_frame_folder_path):
                os.makedirs(group_frame_folder_path)
            log('>>>>({}/{}) {}'.format(id_frame+1, len(group_df), group_frame_folder_name))

            # TIRadar
            sensor = 'TIRadar'
            tmp1 = read_json(os.path.join(root_calibration, 'TIRadar_to_IRayCamera_extrinsic.json'))
            tmp2 = read_json(os.path.join(root_calibration, 'TIRadar_to_LeopardCamera0_extrinsic.json'))
            tmp3 = read_json(os.path.join(root_calibration, 'TIRadar_to_LeopardCamera1_extrinsic.json'))
            json_data = {
                'TIRadar_to_IRayCamera_extrinsic': tmp1['RT'],
                'TIRadar_to_LeopardCamera0_extrinsic': tmp2['RT'],
                'TIRadar_to_LeopardCamera1_extrinsic': tmp3['RT'],
            }
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, json_data)

            # OCULiiRadar
            sensor = 'OCULiiRadar'
            tmp1 = read_json(os.path.join(root_calibration, 'OCULiiRadar_to_IRayCamera_extrinsic.json'))
            tmp2 = read_json(os.path.join(root_calibration, 'OCULiiRadar_to_LeopardCamera0_extrinsic.json'))
            tmp3 = read_json(os.path.join(root_calibration, 'OCULiiRadar_to_LeopardCamera1_extrinsic.json'))
            json_data = {
                'OCULiiRadar_to_IRayCamera_extrinsic': tmp1['RT'],
                'OCULiiRadar_to_LeopardCamera0_extrinsic': tmp2['RT'],
                'OCULiiRadar_to_LeopardCamera1_extrinsic': tmp3['RT'],
            }
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, json_data)

            # VelodyneLidar
            sensor = 'VelodyneLidar'
            tmp1 = read_json(os.path.join(root_calibration, 'VelodyneLidar_to_LeopardCamera0_extrinsic.json'))
            tmp2 = read_json(os.path.join(root_calibration, 'VelodyneLidar_to_LeopardCamera1_extrinsic.json'))
            json_data = {
                'VelodyneLidar_to_LeopardCamera0_extrinsic': tmp1['RT'],
                'VelodyneLidar_to_LeopardCamera1_extrinsic': tmp2['RT'],
            }
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, json_data)

            # IRayCamera
            sensor = 'IRayCamera'
            json_data = read_json(os.path.join(root_calibration, 'IRayCamera_intrinsic.json'))
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, json_data)

            # LeopardCamera0
            sensor = 'LeopardCamera0'
            json_data = read_json(os.path.join(root_calibration, 'LeopardCamera0_intrinsic.json'))
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, json_data)

            # LeopardCamera1
            sensor = 'LeopardCamera1'
            json_data = read_json(os.path.join(root_calibration, 'LeopardCamera1_intrinsic.json'))
            tmp1 = read_json(os.path.join(root_calibration, 'LeopardCamera1_to_LeopardCamera0_extrinsic.json'))
            json_data['LeopardCamera1_to_LeopardCamera0_extrinsic'] = tmp1['RT']
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, json_data)

            # MEMS
            sensor = 'MEMS'
            ts = group_df['{}_ts'.format(sensor)].iloc[id_frame]
            file_path = group_df['{}_files_path'.format(sensor)].iloc[id_frame]
            file_name = os.path.basename(file_path)
            suffix = '.{}'.format(file_name.split('.')[-1])
            sensor_path = os.path.join(group_frame_folder_path, sensor)
            if not os.path.exists(sensor_path):
                os.makedirs(sensor_path)
            # read json
            json_data = read_json(file_path)
            json_data['timestamp'] = file_name.replace(suffix, '')
            # save json
            file_name_new = '{:.3f}{}'.format(ts, suffix)
            file_path_new = os.path.join(sensor_path, file_name_new)
            save_dict_as_json(file_path_new, json_data)



    print('done')

def generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, json_data):
    ts = group_df['{}_ts'.format(sensor)].iloc[id_frame]
    file_path = group_df['{}_files_path'.format(sensor)].iloc[id_frame]
    file_name = os.path.basename(file_path)
    suffix = '.{}'.format(file_name.split('.')[-1])
    sensor_path = os.path.join(group_frame_folder_path, sensor)
    if not os.path.exists(sensor_path):
        os.makedirs(sensor_path)

    file_name_new = '{:.3f}{}'.format(ts, suffix)
    file_path_new = os.path.join(sensor_path, file_name_new)
    shutil.copy(file_path, file_path_new)

    json_data['timestamp'] = file_name.replace(suffix, '')
    json_path = os.path.join(sensor_path, file_name_new.replace(suffix, '.json'))
    save_dict_as_json(json_path, json_data)
    log_GREEN('    generate {} folder completely '.format(sensor))


if __name__ == '__main__':
    main()

