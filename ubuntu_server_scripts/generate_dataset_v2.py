import os
import shutil
import sys
import argparse
import glob

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import pandas as pd

from Camera.timestamp_selector import DataFolder as CDF
from MEMS.timestamp_selector import DataFolder as MDF
from OCUliiRadar.timestamp_selector import DataFolder as ODF
from TIRadar.timestamp_selector import DataFolder as TDF
from VelodyneLidar.timestamp_selector import DataFolder as VDF
from utils import *
from dataset_v2 import load_json, save_dict_as_json


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

def add_calibres_to_dict(json_data, root_calibration, day, json_name, key=None):
    if os.path.exists(root_calibration):
        res_path = ''
        if day in ['20221217', '20221219', '20221220', '20221221']:
            res_path = os.path.join(root_calibration, 'code', 'zhoushan_20221217_20221221', 'results', json_name)
        elif day in ['20221223', '20221224']:
            res_path = os.path.join(root_calibration, 'code', 'yantai_20221223_20221226', 'results', json_name)
        else:
            log_YELLOW('Do not have calibration results for {}'.format(day))

        if os.path.exists(res_path):
            calib_res = load_json(res_path)
            if key is not None:
                if key in calib_res.keys():
                    json_data[json_name.replace('.json', '')] = calib_res[key]
                else:
                    log_YELLOW('{} not in {}'.format(key, res_path))
            else:
                for item in calib_res.keys():
                    json_data[item] = calib_res[item]
    return json_data

def generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, sensor_info):
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

    sensor_info['timestamp'] = file_name.replace(suffix, '')
    json_path = os.path.join(sensor_path, file_name_new.replace(suffix, '.json'))
    save_dict_as_json(json_path, sensor_info)
    log_GREEN('    generate {} folder completely '.format(sensor))

def add_calibmat(sensor, root_calibration, day, group_df, id_frame, group_frame_folder_path):
    mode_name = group_df['TIRadar_group'].iloc[id_frame].split('_')[2]

    mat_path = ''
    if day in ['20221217', '20221219', '20221220', '20221221']:
        mat_path = glob.glob(os.path.join(root_calibration, 'code', 'zhoushan_20221217_20221221', 'results', '{}*.mat'.format(mode_name)))[0]
    elif day in ['20221223', '20221224']:
        mat_path = glob.glob(os.path.join(root_calibration, 'code', 'yantai_20221223_20221226', 'results', '{}*.mat'.format(mode_name)))[0]
    else:
        log_YELLOW('Do not have calibmat results for {} {}'.format(day, mode_name))

    if os.path.exists(mat_path):
        sensor_path = os.path.join(group_frame_folder_path, sensor)
        shutil.copy(mat_path, sensor_path)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, help='exp path')
    parser.add_argument('--calib_path', type=str, default='/mnt/Dataset/sensors_calibration_v2', help='calibration results path')
    parser.add_argument('--output_path', type=str, default='/mnt/Dataset/ourDataset_v2', help='output path for dataset_v2')
    parser.add_argument('--time_error_thred', type=float, default=0.05, help='abs min error between TIRadar and other sensors')
    parser.add_argument('--min_cnt_frames', type=int, default=60, help='min number of frames in a group')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    root_data = args.data_path
    log('ready to process {}'.format(root_data))

    root_TIRadar = os.path.join(root_data, 'TIRadar_npz')
    root_IRayCamera = os.path.join(root_data, 'IRayCamera')
    root_LeopardCamera0 = os.path.join(root_data, 'LeopardCamera0')
    root_LeopardCamera1 = os.path.join(root_data, 'LeopardCamera1')
    root_OCULiiRadar = os.path.join(root_data, 'OCULiiRadar_pcd')
    root_MEMS = os.path.join(root_data, 'MEMS_json')
    root_VelodyneLidar = os.path.join(root_data, 'VelodyneLidar_pcd')

    root_calibration = args.calib_path

    root_output = args.output_path
    if root_output is not None:
        if not os.path.exists(root_output):
            os.mkdir(root_output)
            log('create {}'.format(root_output))

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

    error_thred = args.time_error_thred

    # filter by time_error with TIRadar
    mask = np.logical_or(
        np.logical_or(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        np.abs(df['IRayCamera_error'].values) > error_thred,
                        np.abs(df['LeopardCamera0_error'].values) > error_thred,
                    ),
                    np.abs(df['LeopardCamera1_error'].values) > error_thred,
                ),
                np.abs(df['MEMS_error'].values) > error_thred,
            ),
            np.abs(df['OCULiiRadar_error'].values) > error_thred,
        ),
        np.abs(df['VelodyneLidar_error'].values) > error_thred,
    )
    # df_illegal = df[mask]
    df_legal = df[np.logical_not(mask)]

    # grouping
    v_loc = df_legal.index.values
    cut_points = np.where((v_loc[1:] - v_loc[:-1]) > 1)[0]
    continue_group = np.zeros((len(df_legal, )), dtype='int32')
    for cut_point in cut_points:
        continue_group[cut_point+1:] += 1
    df_legal.insert(loc=0, column='continue_group', value=continue_group)

    min_cnt_frames = args.min_cnt_frames
    cnt_group_frames = []
    groups_info = []
    res_groupby = df_legal.groupby(['continue_group', 'TIRadar_group'])

    # normal mode group filter by min_cnt_frames
    for group_name, group_df in res_groupby:
        cnt_frames = len(group_df)
        day = group_name[1].split('_')[0]
        mode_name = group_name[1].split('_')[2]

        if cnt_frames < min_cnt_frames and mode_name != 'mixmode':
            log_YELLOW(
                'Skip continue_group={}, TIRadar_group={}, the number of frames is {}(<{})'.format(group_name[0], group_name[1], cnt_frames, min_cnt_frames)
            )
            continue
        cnt_group_frames.append(cnt_frames)
        groups_info.append(
            {
                'group_df': group_df,
                'day': day,
                'mode_name': mode_name
            }
        )

    # generate group folder and files
    cnt_groups = len(groups_info)
    for id_group in range(cnt_groups):
        log('=' * 100)

        day = groups_info[id_group]['day']
        mode_name = groups_info[id_group]['mode_name']
        group_df = groups_info[id_group]['group_df']
        cnt_frames = len(group_df)

        group_folder_name = '{}_group{:>04d}_{}_{}frames'.format(day, id_group, mode_name, len(group_df))
        group_folder_path = os.path.join(root_output, group_folder_name)
        # create group_folder
        if not os.path.exists(group_folder_path):
            os.mkdir(group_folder_path)
            log('create {}'.format(group_folder_path))

        for id_frame in range(len(group_df)):
            group_frame_folder_name = 'frame{:>04d}'.format(id_frame)
            group_frame_folder_path = os.path.join(group_folder_path, group_frame_folder_name)
            if not os.path.exists(group_frame_folder_path):
                os.makedirs(group_frame_folder_path)
            log(
                '>>>> ({}/{}){} | ({}/{}){}'.format(
                    id_group+1, cnt_groups, group_folder_name,
                    id_frame+1, cnt_frames, group_frame_folder_name
                )
            )

            # TIRadar
            sensor = 'TIRadar'
            sensor_info = {}
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'TIRadar_to_IRayCamera_extrinsic.json', 'RT')
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'TIRadar_to_LeopardCamera0_extrinsic.json', 'RT')
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'TIRadar_to_LeopardCamera1_extrinsic.json', 'RT')
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, sensor_info)
            add_calibmat(sensor, root_calibration, day, group_df, id_frame, group_frame_folder_path)

            # OCULiiRadar
            sensor = 'OCULiiRadar'
            sensor_info = {}
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'OCULiiRadar_to_IRayCamera_extrinsic.json', 'RT')
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'OCULiiRadar_to_LeopardCamera0_extrinsic.json', 'RT')
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'OCULiiRadar_to_LeopardCamera1_extrinsic.json', 'RT')
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, sensor_info)

            # VelodyneLidar
            sensor = 'VelodyneLidar'
            sensor_info = {}
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'VelodyneLidar_to_IRayCamera_extrinsic.json', 'RT')
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'VelodyneLidar_to_LeopardCamera0_extrinsic.json', 'RT')
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'VelodyneLidar_to_LeopardCamera1_extrinsic.json', 'RT')
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, sensor_info)

            # IRayCamera
            sensor = 'IRayCamera'
            sensor_info = {}
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'IRayCamera_intrinsic.json')
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, sensor_info)

            # LeopardCamera0
            sensor = 'LeopardCamera0'
            sensor_info = {}
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'LeopardCamera0_intrinsic.json')
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, sensor_info)

            # LeopardCamera1
            sensor = 'LeopardCamera1'
            sensor_info = {}
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'LeopardCamera1_intrinsic.json')
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'LeopardCamera1_to_LeopardCamera0_extrinsic.json', 'RT')
            generate_sensor_folder(sensor, group_df, id_frame, group_frame_folder_path, sensor_info)

            # MEMS
            sensor = 'MEMS'
            ts = group_df['{}_ts'.format(sensor)].iloc[id_frame]
            file_path = group_df['{}_files_path'.format(sensor)].iloc[id_frame]
            file_name = os.path.basename(file_path)
            suffix = '.{}'.format(file_name.split('.')[-1])
            sensor_path = os.path.join(group_frame_folder_path, sensor)
            if not os.path.exists(sensor_path):
                os.makedirs(sensor_path)
            # read MEMS data in json
            sensor_info = load_json(file_path)
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'MEMS_to_Vehicle_extrinsic.json', 'RT')
            sensor_info = add_calibres_to_dict(sensor_info, root_calibration, day, 'MEMS_to_VelodyneLidar_extrinsic.json', 'RT')
            sensor_info['timestamp'] = file_name.replace(suffix, '')
            # save json
            file_name_new = '{:.3f}{}'.format(ts, suffix)
            file_path_new = os.path.join(sensor_path, file_name_new)
            save_dict_as_json(file_path_new, sensor_info)
            log_GREEN('    generate {} folder completely '.format(sensor))

    print('done')


if __name__ == '__main__':
    main()

