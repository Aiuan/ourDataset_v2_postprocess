import os
import shutil
import glob

import numpy as np
import pandas as pd

from Camera.timestamp_selector import DataFolder as CDF
from MEMS.timestamp_selector import DataFolder as MDF
from OCUliiRadar.timestamp_selector import DataFolder as ODF
from TIRadar.timestamp_selector import DataFolder as TDF
from VelodyneLidar.timestamp_selector import DataFolder as VDF

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
    root_TIRadar = 'F:\\20221217\\TIRadar_npz'
    root_IRayCamera = 'F:\\20221217\\IRayCamera'
    root_LeopardCamera0 = 'F:\\20221217\\LeopardCamera0'
    root_LeopardCamera1 = 'F:\\20221217\\LeopardCamera1'
    root_OCULiiRadar = 'F:\\20221217\\OCULiiRadar_pcd'
    root_MEMS = 'F:\\20221217\\MEMS_json'
    root_VelodyneLidar = 'F:\\20221217\\VelodyneLidar_pcd'

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

    for group_name, group_df in df_legal.groupby(['continue_group', 'TIRadar_group']):
        print('done')


    print('done')


if __name__ == '__main__':
    main()

