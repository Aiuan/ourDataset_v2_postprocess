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

    n_row = 2
    n_col = 3

    fig_base = 4
    fig_width = fig_base * n_col
    fig_height = fig_base * n_row
    fig = plt.figure(figsize=(fig_width, fig_height))

    # image
    IRayCamera_png_height, IRayCamera_png_width = IRayCamera_png.shape
    LeopardCamera0_png_height, LeopardCamera0_png_width, _ = LeopardCamera0_png.shape
    LeopardCamera1_png_height, LeopardCamera1_png_width, _ = LeopardCamera1_png.shape

    w_padding = 20
    w0 = w_padding / (IRayCamera_png_width + LeopardCamera0_png_width + LeopardCamera1_png_width + (
                n_col + 1) * w_padding)
    w1 = IRayCamera_png_width / (IRayCamera_png_width + LeopardCamera0_png_width + LeopardCamera1_png_width + (
                n_col + 1) * w_padding)
    w2 = LeopardCamera0_png_width / (
                IRayCamera_png_width + LeopardCamera0_png_width + LeopardCamera1_png_width + (
                    n_col + 1) * w_padding)
    w3 = LeopardCamera1_png_width / (
                IRayCamera_png_width + LeopardCamera0_png_width + LeopardCamera1_png_width + (
                    n_col + 1) * w_padding)
    h1 = 0.5

    ax1 = fig.add_axes((w0, 1 - h1, w1, h1))
    ax1.imshow(IRayCamera_png)
    ax1.axis('off')
    ax1.set_title('IRayCamera\n{}'.format(unix2local(IRayCamera_ts_str)))

    ax2 = fig.add_axes((w0 + w1 + w0, 1 - h1, w2, h1))
    ax2.imshow(LeopardCamera0_png)
    ax2.axis('off')
    ax2.set_title('LeopardCamera0\n{}'.format(unix2local(LeopardCamera0_ts_str)))

    ax3 = fig.add_axes((w0 + w1 + w0 + w2 + w0, 1 - h1, w3, h1))
    ax3.imshow(LeopardCamera1_png)
    ax3.axis('off')
    ax3.set_title('LeopardCamera1\n{}'.format(unix2local(LeopardCamera1_ts_str)))

    # pointcloud
    r = 40
    r_padding = 0
    r0 = r_padding / (6 * r + 4 * r_padding)
    r1 = (2 * r) / (6 * r + 4 * r_padding)
    r2 = r1
    h2 = 0.1

    OCULiiRadar_xyz = pcd_in_zone(OCULiiRadar_pcd, xlim=[-r, r], zlim=[0, 2 * r])
    ax4 = axes3d.Axes3D(fig, (r0, h2, r2, r2))
    ax4.scatter3D(
        OCULiiRadar_xyz[:, 0], OCULiiRadar_xyz[:, 1], OCULiiRadar_xyz[:, 2],
        c=OCULiiRadar_xyz[:, 4], s=1, cmap='jet'
    )
    ax4.set_xlim3d([-r, r])
    ax4.set_zlim3d([0, 2 * r])
    if len(OCULiiRadar_xyz) > 0:
        rate = (OCULiiRadar_xyz[:, 1].max() - OCULiiRadar_xyz[:, 1].min()) / (2 * r)
        if rate > 0:
            ax4.set_box_aspect((1, rate, 1))
    ax4.set_xlabel('x/m')
    ax4.set_ylabel('y/m')
    ax4.set_zlabel('z/m')
    ax4.view_init(elev=0, azim=-90)
    ax4.set_title('OCULiiRadar\n{}'.format(unix2local(OCULiiRadar_ts_str)))

    VelodyneLidar_xyz = pcd_in_zone(VelodyneLidar_pcd, xlim=[-r, r], ylim=[0, 2 * r])
    ax5 = axes3d.Axes3D(fig, (r0 + r1 + r0, h2, r2, r2))
    ax5.scatter3D(
        VelodyneLidar_xyz[:, 0], VelodyneLidar_xyz[:, 1], VelodyneLidar_xyz[:, 2],
        c=VelodyneLidar_xyz[:, 3], s=1, cmap='jet'
    )
    ax5.set_xlim3d([-r, r])
    ax5.set_ylim3d([0, 2 * r])
    if len(VelodyneLidar_xyz) > 0:
        rate = (VelodyneLidar_xyz[:, 2].max() - VelodyneLidar_xyz[:, 2].min()) / (2 * r)
        if rate > 0:
            ax5.set_box_aspect((1, 1, rate))
    ax5.set_xlabel('x/m')
    ax5.set_ylabel('y/m')
    ax5.set_zlabel('z/m')
    ax5.view_init(elev=80, azim=-90)
    ax5.set_title('VelodyneLidar\n{}'.format(unix2local(VelodyneLidar_ts_str)))

    TIRadar_xyz = pcd_in_zone(TIRadar_pcd, xlim=[-r, r], ylim=[0, 2 * r])
    ax6 = axes3d.Axes3D(fig, (r0 + r1 + r0 + r1 + r0, h2, r2, r2))
    ax6.scatter3D(
        TIRadar_xyz[:, 0], TIRadar_xyz[:, 1], TIRadar_xyz[:, 2],
        c=TIRadar_xyz[:, 4], s=1, cmap='jet'
    )
    ax6.set_xlim3d([-r, r])
    ax6.set_ylim3d([0, 2 * r])
    if len(TIRadar_xyz) > 0:
        rate = (TIRadar_xyz[:, 2].max() - TIRadar_xyz[:, 2].min()) / (2 * r)
        if rate > 0:
            ax6.set_box_aspect((1, 1, rate))
    ax6.set_xlabel('x/m')
    ax6.set_ylabel('y/m')
    ax6.set_zlabel('z/m')
    ax6.view_init(elev=80, azim=-90)
    ax6.set_title('TIRadar\n{}'.format(unix2local(TIRadar_ts_str)))

    plt.show()

    print('done')

if __name__ == '__main__':
    main()

