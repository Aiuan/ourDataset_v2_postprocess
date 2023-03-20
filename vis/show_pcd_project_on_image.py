import os
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import matplotlib.pyplot as plt

from dataset_v2 import *

def show_pcd_project_on_image(
        pcd, image, camera_intrinsic, pcd2camera_extrinsic,
        pcd_zone=None, pcd_ts_str=None, image_ts_str=None, radial_distortion=None, tangential_distortion=None
):
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

    # image
    if (radial_distortion is not None) and (tangential_distortion is not None):
        image = undistort_image(image, camera_intrinsic, radial_distortion, tangential_distortion)

    ax.imshow(image)

    # pcd
    if pcd_zone is not None:
        xyz = pcd_in_zone(
            pcd, xlim=[pcd_zone[0], pcd_zone[1]],
            ylim=[pcd_zone[2], pcd_zone[3]], zlim=[pcd_zone[4], pcd_zone[5]]
        ).T
    else:
        xyz = pcd_in_zone(pcd).T
    xyz1 = np.vstack((xyz[:3, :], np.ones((1, xyz.shape[1]), dtype=np.float32)))
    project_matrix = np.matmul(camera_intrinsic, pcd2camera_extrinsic[:3, :])
    uvz = np.matmul(project_matrix, xyz1)
    uv1 = uvz / uvz[2, :]
    uv = uv1[:2, :]
    uv = np.round(uv)
    uv = np.vstack((uv, xyz[3:, :]))

    h, w, _ = image.shape
    mask_in_image = np.logical_and(
        np.logical_and(uv[0, :] > 0, uv[0, :] < w),
        np.logical_and(uv[1, :] > 0, uv[1, :] < h)
    )
    uv_in_image = uv[:, mask_in_image]

    ax.scatter(
        uv_in_image[0, :], uv_in_image[1, :],
        s=1, c=uv_in_image[2, :], cmap='jet'
    )

    if (pcd_ts_str is not None) and (image_ts_str is not None):
        ts_offset = float(pcd_ts_str) - float(image_ts_str)
        if ts_offset >= 0:
            title = 'pcd_timestamp: {} (+{:.0f}ms)\nimage_timestamp: {}'.format(pcd_ts_str, abs(ts_offset*1000), image_ts_str)
        else:
            title = 'pcd_timestamp: {} (-{:.0f}ms)\nimage_timestamp: {}'.format(pcd_ts_str, abs(ts_offset*1000), image_ts_str)
        plt.title(title, loc='center')



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

    show_pcd_project_on_image(
        VelodyneLidar_pcd,
        LeopardCamera0_png,
        camera_intrinsic=np.array(LeopardCamera0_json['intrinsic_matrix']),
        pcd2camera_extrinsic=np.array(VelodyneLidar_json['VelodyneLidar_to_LeopardCamera0_extrinsic']),
        pcd_zone=[-40, 40, 0, 40, -20, 20],
        pcd_ts_str=VelodyneLidar_ts_str,
        image_ts_str=LeopardCamera0_ts_str,
        radial_distortion=np.array(LeopardCamera0_json['radial_distortion']),
        tangential_distortion=np.array(LeopardCamera0_json['tangential_distortion'])
    )

    show_pcd_project_on_image(
        VelodyneLidar_pcd,
        LeopardCamera1_png,
        camera_intrinsic=np.array(LeopardCamera1_json['intrinsic_matrix']),
        pcd2camera_extrinsic=np.array(VelodyneLidar_json['VelodyneLidar_to_LeopardCamera1_extrinsic']),
        pcd_zone=[-40, 40, 0, 40, -20, 20],
        pcd_ts_str=VelodyneLidar_ts_str,
        image_ts_str=LeopardCamera1_ts_str,
        radial_distortion=np.array(LeopardCamera1_json['radial_distortion']),
        tangential_distortion=np.array(LeopardCamera1_json['tangential_distortion'])
    )

    # show_pcd_project_on_image(
    #     VelodyneLidar_pcd,
    #     IRayCamera_png,
    #     camera_intrinsic=np.array(IRayCamera_json['intrinsic_matrix']),
    #     pcd2camera_extrinsic=np.array(VelodyneLidar_json['VelodyneLidar_to_IRayCamera_extrinsic']),
    #     pcd_zone=[-40, 40, 0, 40, -20, 20],
    #     pcd_ts_str=VelodyneLidar_ts_str,
    #     image_ts_str=IRayCamera_ts_str,
    #     radial_distortion=np.array(IRayCamera_json['radial_distortion']),
    #     tangential_distortion=np.array(IRayCamera_json['tangential_distortion'])
    # )



    plt.show()
    print('done')


if __name__ == '__main__':
    main()

