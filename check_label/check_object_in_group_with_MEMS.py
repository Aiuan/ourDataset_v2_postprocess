'''
    L：表示目标框长度在这一组数据中方差较大
    W：表示目标框长度在这一组数据中方差较大
    H：表示目标框长度在这一组数据中方差较大
    Sx：表示在motion标注为0的情况下，即目标静止，通过IMU补偿自身位置偏移后，计算前后两帧目标框x值差，大于阈值
    Sy：表示在motion标注为0的情况下，即目标静止，通过IMU补偿自身位置偏移后，计算前后两帧目标框y值差，大于阈值
    Sz：表示在motion标注为0的情况下，即目标静止，通过IMU补偿自身位置偏移后，计算前后两帧目标框z值差，大于阈值
    Sa：表示在motion标注为0的情况下，即目标静止，通过IMU补偿自身位置偏移后，计算前后两帧目标框alpha值差，大于阈值
    Da：表示在motion标注为1的情况下，即目标运动，通过IMU补偿自身位置偏移后，计算前后两帧目标框xyz值差，假设两帧间匀速直线运动，计算出一个alpha_pred，abs（alpha_pred-当前帧alpha）大于阈值
'''

import glob
import os
import sys
import argparse

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import pandas as pd

from dataset_v2 import log, load_json

def pos2ecef(lon, lat, h):
    # WGS84长半轴
    a = 6378137
    # WGS84椭球扁率
    f = 1 / 298.257223563
    # WGS84短半轴
    b = a * (1 - f)
    # WGS84椭球第一偏心率
    e = np.sqrt(a * a - b * b) / a
    # WGS84椭球面卯酉圈的曲率半径
    N = a / np.sqrt(1 - e * e * np.sin(lat * np.pi / 180) * np.sin(lat * np.pi / 180))
    x_ecef = (N + h) * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    y_ecef = (N + h) * np.cos(lat * np.pi / 180) * np.sin(lon * np.pi / 180)
    z_ecef = (N * (1 - (e * e)) + h) * np.sin(lat * np.pi / 180)

    return x_ecef, y_ecef, z_ecef

def pos2enu(lon, lat, h, lon_ref, lat_ref, h_ref):
    x_ecef, y_ecef, z_ecef = pos2ecef(lon, lat, h)
    x_ecef_ref, y_ecef_ref, z_ecef_ref = pos2ecef(lon_ref, lat_ref, h_ref)

    offset_x, offset_y, offset_z = x_ecef - x_ecef_ref, y_ecef - y_ecef_ref, z_ecef - z_ecef_ref

    sinLon = np.sin(lon_ref * np.pi / 180)
    cosLon = np.cos(lon_ref * np.pi / 180)
    sinLat = np.sin(lat_ref * np.pi / 180)
    cosLat = np.cos(lat_ref * np.pi / 180)

    x_enu = -1 * sinLon * offset_x + cosLon * offset_y
    y_enu = -1 * sinLat * cosLon * offset_x - 1 * sinLat * sinLon * offset_y + cosLat * offset_z
    z_enu = cosLat * cosLon * offset_x + cosLat * sinLon * offset_y + sinLat * offset_z

    return x_enu, y_enu, z_enu

def cosd(degree):
    return np.cos(degree / 180 * np.pi)

def sind(degree):
    return np.sin(degree / 180 * np.pi)

def b2n(pitch, roll, yaw, tx, ty, tz):
    # Z-X-Y
    return np.array([
        [cosd(roll)*cosd(yaw)-sind(pitch)*sind(roll)*sind(yaw), -cosd(pitch)*sind(yaw), cosd(yaw)*sind(roll)+cosd(roll)*sind(pitch)*sind(yaw), tx],
        [cosd(roll)*sind(yaw) + cosd(yaw)*sind(pitch)*sind(roll), cosd(pitch)*cosd(yaw), sind(roll)*sind(yaw)-cosd(roll)*cosd(yaw)*sind(pitch), ty],
        [-cosd(pitch)*sind(roll), sind(pitch), cosd(pitch)*cosd(roll), tz],
        [0, 0, 0, 1]
    ])

def get_praxyz(mems, mems_ref):
    pitch = mems['msg_ins']['pitch']
    roll = mems['msg_ins']['roll']
    azimuth = mems['msg_ins']['azimuth']

    lat_ref = mems_ref['msg_ins']['latitude_N']
    lon_ref = mems_ref['msg_ins']['longitude_E']
    h_ref = mems_ref['msg_ins']['height']
    lat = mems['msg_ins']['latitude_N']
    lon = mems['msg_ins']['longitude_E']
    h = mems['msg_ins']['height']
    x, y, z = pos2enu(lon, lat, h, lon_ref, lat_ref, h_ref)

    return pitch, roll, azimuth, x, y, z

def calculate_pose_relative(mems, mems0):
    pitch0, roll0, azimuth0, x0, y0, z0 = get_praxyz(mems0, mems0)
    yaw0 = 360 - azimuth0 if azimuth0 > 180 else -azimuth0
    pose0 = b2n(pitch0, roll0, yaw0, x0, y0, z0)

    pitch, roll, azimuth, x, y, z = get_praxyz(mems, mems0)
    yaw = 360 - azimuth if azimuth > 180 else -azimuth
    pose = b2n(pitch, roll, yaw, x, y, z)

    # coordinate = pose_relative * coordinate_0
    pose_relative = np.matmul(np.linalg.inv(pose), pose0)

    return pose_relative

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dataset', type=str, default='/mnt/Dataset/ourDataset_v2_label', help='MEMS path')
    parser.add_argument('--root_label', type=str, default='/mnt/Dataset/labels_for_checking2', help='labelroot path')
    parser.add_argument('--label_foldername', type=str, default='', help='label folder name')
    parser.add_argument('--output_foldername', type=str, default='runs2/object_MEMS', help='output folder name')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    root_dataset = args.root_dataset
    root_label = args.root_label
    label_foldername = args.label_foldername
    output_foldername = args.output_foldername

    if label_foldername != '':
        label_foldernames = [label_foldername]
    else:
        label_foldernames = os.listdir(root_label)
        label_foldernames.sort()

    root_output = os.path.join(CURRENT_ROOT, output_foldername)
    if not os.path.exists(root_output):
        os.makedirs(root_output)
        log('create {}'.format(root_output))

    for label_foldername in label_foldernames:
        label_folder = os.path.join(root_label, label_foldername)
        groupnames = os.listdir(label_folder)
        groupnames.sort()
        for i, groupname in enumerate(groupnames):
            log('=' * 100)
            log('{}/{} {}'.format(i + 1, len(groupnames), groupname))

            output_folder = os.path.join(root_output, '{}-{}'.format(label_foldername, groupname))
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
                log('create {}'.format(output_folder))

            df_group_path = os.path.join(output_folder, 'df_group.xlsx')
            if os.path.exists(df_group_path):
                df_group = pd.read_excel(df_group_path, index_col=0)
            else:
                df_group = pd.DataFrame()
                group_folder = os.path.join(label_folder, groupname)
                framenames = os.listdir(group_folder)
                framenames = [framename.split('.')[0] for framename in framenames]
                framenames.sort()
                for j, framename in enumerate(framenames):
                    log('{}/{} {}'.format(j + 1, len(framenames), framename))

                    labels = load_json(os.path.join(group_folder, '{}.json'.format(framename)))
                    df_frame = pd.DataFrame(labels)

                    df_frame['framename'] = framename

                    # calculate the pose change relative to the first frame
                    mems0 = load_json(glob.glob(os.path.join(root_dataset, groupname, framenames[0], 'MEMS', '*.json'))[0])
                    mems = load_json(glob.glob(os.path.join(root_dataset, groupname, framename, 'MEMS', '*.json'))[0])
                    pose_relative = calculate_pose_relative(mems, mems0)
                    for p in range(pose_relative.shape[0]):
                        for q in range(pose_relative.shape[1]):
                            df_frame['pose{}{}'.format(p, q)] = pose_relative[p, q]

                    df_group = pd.concat((df_group, df_frame))
                df_group['groupname'] = groupname
                df_group.to_excel(df_group_path)

            # check each object
            for object_id, df_object in df_group.groupby(['object_id']):
                print('object_id = {}'.format(object_id))

                msg = ''

                # judge object's lwh
                if df_object['l'].std() >= 0.1:
                    msg = msg + 'L'
                if df_object['w'].std() >= 0.1:
                    msg = msg + 'W'
                if df_object['h'].std() >= 0.1:
                    msg = msg + 'H'


                for k in range(df_object.shape[0]):
                    if k == 0:
                        df_object['x_last_transfer'] = np.nan
                        df_object['delta_x'] = np.nan
                        df_object['y_last_transfer'] = np.nan
                        df_object['delta_y'] = np.nan
                        df_object['z_last_transfer'] = np.nan
                        df_object['delta_z'] = np.nan
                        df_object['alpha_last_transfer'] = np.nan
                        df_object['delta_alpha'] = np.nan
                        df_object['alpha_pred'] = np.nan
                        df_object['error_alpha_pred'] = np.nan
                        continue
                    else:
                        pose_last = []
                        for p in range(4):
                            for q in range(4):
                                pose_last.append(df_object['pose{}{}'.format(p, q)].iloc[k-1])
                        pose_last = np.array(pose_last).reshape((4, 4))
                        xyz1_last = np.array([df_object['x'].iloc[k-1], df_object['y'].iloc[k-1], df_object['z'].iloc[k-1], 1])
                        alpha_last = df_object['alpha'].iloc[k-1]

                        pose = []
                        for p in range(4):
                            for q in range(4):
                                pose.append(df_object['pose{}{}'.format(p, q)].iloc[k])
                        pose = np.array(pose).reshape((4, 4))
                        xyz1 = np.array([df_object['x'].iloc[k], df_object['y'].iloc[k], df_object['z'].iloc[k], 1])
                        alpha = df_object['alpha'].iloc[k]

                        xyz1_last_transfer = np.matmul(
                            np.matmul(pose, np.linalg.inv(pose_last)), xyz1_last.reshape((-1, 1))
                        ).reshape((-1))
                        df_object['x_last_transfer'].iloc[k] = xyz1_last_transfer[0]
                        df_object['delta_x'].iloc[k] = xyz1[0] - xyz1_last_transfer[0]
                        df_object['y_last_transfer'].iloc[k] = xyz1_last_transfer[1]
                        df_object['delta_y'].iloc[k] = xyz1[1] - xyz1_last_transfer[1]
                        df_object['z_last_transfer'].iloc[k] = xyz1_last_transfer[2]
                        df_object['delta_z'].iloc[k] = xyz1[2] - xyz1_last_transfer[2]

                        # alpha
                        ori = np.array([1, 0, 0])
                        ori_last = np.matmul(
                            np.array([
                                [np.cos(alpha_last), -np.sin(alpha_last), 0],
                                [np.sin(alpha_last), np.cos(alpha_last), 0],
                                [0, 0, 1]
                            ]),
                            ori.reshape((-1, 1))
                        ).reshape((-1))
                        ori_last_transfer = np.matmul(
                            np.matmul(pose[:3, :3], np.linalg.inv(pose_last[:3, :3])), ori_last.reshape((-1, 1))
                        ).reshape((-1))
                        if ori_last_transfer[0] <= 0 and ori_last_transfer[1] >= 0:
                            alpha_last_transfer = np.arctan(ori_last_transfer[1] / ori_last_transfer[0]) + np.pi
                        elif ori_last_transfer[0] <= 0 and ori_last_transfer[1] <= 0:
                            alpha_last_transfer = np.arctan(ori_last_transfer[1] / ori_last_transfer[0]) - np.pi
                        else:
                            alpha_last_transfer = np.arctan(ori_last_transfer[1] / ori_last_transfer[0])
                        df_object['alpha_last_transfer'].iloc[k] = alpha_last_transfer
                        df_object['delta_alpha'].iloc[k] = alpha - alpha_last_transfer

                        # alpha_pred
                        if df_object['delta_x'].iloc[k] <= 0 and df_object['delta_y'].iloc[k] >= 0:
                            df_object['alpha_pred'].iloc[k] = np.arctan(
                                df_object['delta_y'].iloc[k] / df_object['delta_x'].iloc[k]
                            ) + np.pi
                        elif df_object['delta_x'].iloc[k] <= 0 and df_object['delta_y'].iloc[k] <= 0:
                            df_object['alpha_pred'].iloc[k] = np.arctan(
                                df_object['delta_y'].iloc[k] / df_object['delta_x'].iloc[k]
                            ) - np.pi
                        else:
                            df_object['alpha_pred'].iloc[k] = np.arctan(
                                df_object['delta_y'].iloc[k] / df_object['delta_x'].iloc[k]
                            )
                        df_object['error_alpha_pred'].iloc[k] = np.abs(df_object['alpha_pred'].iloc[k] - alpha)

                        if df_object['motion'].iloc[k] == 0:
                            # judge static object
                            if np.abs(df_object['delta_x'].iloc[k]) >= 0.5:
                                if 'Sx' not in msg:
                                    msg = msg + 'Sx'
                            if np.abs(df_object['delta_y'].iloc[k]) >= 0.5:
                                if 'Sy' not in msg:
                                    msg = msg + 'Sy'
                            if np.abs(df_object['delta_z'].iloc[k]) >= 0.5:
                                if 'Sz' not in msg:
                                    msg = msg + 'Sz'

                            if np.abs(df_object['delta_alpha'].iloc[k]) >= 0.05:
                                if 'Sa' not in msg:
                                    msg = msg + 'Sa'
                        else:
                            # judge dynamic object
                            if df_object['error_alpha_pred'].iloc[k] >= 0.2:
                                if 'Da' not in msg:
                                    msg = msg + 'Da'

                df_object.to_excel(os.path.join(output_folder, 'object{}_{}.xlsx'.format(object_id, msg)))


    print('done')


if __name__ == '__main__':
    main()

