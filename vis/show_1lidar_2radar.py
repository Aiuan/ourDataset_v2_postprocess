import os
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from dataset_v2 import *


def main():
    root_frame = '/mnt/Dataset/ourDataset_v2/20221217_group0000_mode1_279frames/frame0000'

    frame = Frame(root_frame)

    IRayCamera_json = frame.get_sensor_data('IRayCamera_json')
    IRayCamera_ts_str = IRayCamera_json['timestamp']

    IRayCamera_png = frame.get_sensor_data('IRayCamera_png')

    LeopardCamera0_json = frame.get_sensor_data('LeopardCamera0_json')
    LeopardCamera0_ts_str = LeopardCamera0_json['timestamp']
    LeopardCamera0_png = frame.get_sensor_data('LeopardCamera0_png')

    LeopardCamera1_json = frame.get_sensor_data('LeopardCamera1_json')
    LeopardCamera1_ts_str = LeopardCamera1_json['timestamp']
    LeopardCamera1_png = frame.get_sensor_data('LeopardCamera1_png')

    VelodyneLidar_json = frame.get_sensor_data('VelodyneLidar_json')
    VelodyneLidar_ts_str = VelodyneLidar_json['timestamp']
    VelodyneLidar_pcd = frame.get_sensor_data('VelodyneLidar_pcd')

    OCULiiRadar_json = frame.get_sensor_data('OCULiiRadar_json')
    OCULiiRadar_ts_str = OCULiiRadar_json['timestamp']
    OCULiiRadar_pcd = frame.get_sensor_data('OCULiiRadar_pcd')

    TIRadar_json = frame.get_sensor_data('TIRadar_json')
    TIRadar_ts_str = TIRadar_json['timestamp']
    TIRadar_pcd = frame.get_sensor_data('TIRadar_pcd')
    TIRadar_heatmapBEV = frame.get_sensor_data('TIRadar_heatmapBEV')

    # pointcloud
    r = 40

    fig = plt.figure()
    ax = axes3d.Axes3D(fig)

    VelodyneLidar_xyz = pcd_in_zone(VelodyneLidar_pcd, xlim=[-r, r], ylim=[0, 2 * r])
    ax.scatter3D(
        VelodyneLidar_xyz[:, 0], VelodyneLidar_xyz[:, 1], VelodyneLidar_xyz[:, 2], c='black', s=1, marker='.'
    )

    VelodyneLidar_to_LeopardCamera0_extrinsic = np.array(VelodyneLidar_json['VelodyneLidar_to_LeopardCamera0_extrinsic'])
    OCULiiRadar_to_LeopardCamera0_extrinsic = np.array(OCULiiRadar_json['OCULiiRadar_to_LeopardCamera0_extrinsic'])
    OCULiiRadar_to_VelodyneLidar_extrinsic = np.matmul(np.linalg.inv(VelodyneLidar_to_LeopardCamera0_extrinsic), OCULiiRadar_to_LeopardCamera0_extrinsic)
    OCULiiRadar_pcd_in_VelodyneLidar_coordinate = pcd_transform(OCULiiRadar_pcd, OCULiiRadar_to_VelodyneLidar_extrinsic)
    OCULiiRadar_xyz = pcd_in_zone(OCULiiRadar_pcd_in_VelodyneLidar_coordinate, xlim=[-r, r], zlim=[0, 2 * r])
    ax.scatter3D(
        OCULiiRadar_xyz[:, 0], OCULiiRadar_xyz[:, 1], OCULiiRadar_xyz[:, 2],
        c=OCULiiRadar_xyz[:, 4], s=4, marker='^',  cmap='jet'
    )
    
    TIRadar_to_LeopardCamera0_extrinsic = np.array(TIRadar_json['TIRadar_to_LeopardCamera0_extrinsic'])
    TIRadar_to_VelodyneLidar_extrinsic = np.matmul(np.linalg.inv(VelodyneLidar_to_LeopardCamera0_extrinsic), TIRadar_to_LeopardCamera0_extrinsic)
    TIRadar_pcd_in_VelodyneLidar_coordinate = pcd_transform(TIRadar_pcd, TIRadar_to_VelodyneLidar_extrinsic)
    TIRadar_xyz = pcd_in_zone(TIRadar_pcd_in_VelodyneLidar_coordinate, xlim=[-r, r], ylim=[0, 2 * r])
    ax.scatter3D(
        TIRadar_xyz[:, 0], TIRadar_xyz[:, 1], TIRadar_xyz[:, 2],
        c=TIRadar_xyz[:, 4], s=4, marker='s', cmap='jet'
    )

    if len(VelodyneLidar_xyz) > 0:
        rate = (VelodyneLidar_xyz[:, 2].max() - VelodyneLidar_xyz[:, 2].min()) / (2 * r)
        if rate > 0:
            ax.set_box_aspect((1, 1, rate))

    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([0, 2 * r])
    ax.set_xlabel('x/m')
    ax.set_ylabel('y/m')
    ax.set_zlabel('z/m')
    ax.view_init(elev=90, azim=-90)

    title = 'VelodyneLidar: {} (+0ms)\n'.format(unix2local(VelodyneLidar_ts_str))
    offset1 = float(OCULiiRadar_ts_str) - float(VelodyneLidar_ts_str)
    if offset1 >= 0:
        title = title + 'OCULiiRadar: {} (+{:.0f}ms)\n'.format(unix2local(OCULiiRadar_ts_str), abs(offset1 * 1000))
    else:
        title = title + 'OCULiiRadar: {} (-{:.0f}ms)\n'.format(unix2local(OCULiiRadar_ts_str), abs(offset1 * 1000))
    offset2 = float(TIRadar_ts_str) - float(VelodyneLidar_ts_str)
    if offset2 >= 0:
        title = title + 'TIRadar: {} (+{:.0f}ms)'.format(unix2local(TIRadar_ts_str), abs(offset2 * 1000))
    else:
        title = title + 'TIRadar: {} (-{:.0f}ms)'.format(unix2local(TIRadar_ts_str), abs(offset2 * 1000))

    ax.set_title(title)

    plt.show()

    print('done')

if __name__ == '__main__':
    main()

