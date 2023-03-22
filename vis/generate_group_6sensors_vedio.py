import os
import sys
import argparse

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from dataset_v2 import *

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dataset', type=str, default='/mnt/Dataset/ourDataset_v2', help='ourDataset_v2 path')
    parser.add_argument('--filter_keys', type=str, nargs='*', default=[], help='keys for filter')
    parser.add_argument('--root_output', type=str, default='/mnt/Dataset/ourDataset_v2_vis/vis_6sensors', help='output path')
    parser.add_argument('--show', type=bool, default=False, help='dynamic show or not')
    parser.add_argument('--cover', type=bool, default=True, help='cover or not')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    root_dataset = args.root_dataset
    filter_keys = args.filter_keys
    root_output = args.root_output
    show = args.show
    cover = args.cover

    if not os.path.exists(root_output):
        os.makedirs(root_output)
        log_GREEN('create {}'.format(root_output))

    w_vedio = 1920
    h_vedio = 1080
    dpi_vedio = 100
    fig = plt.figure(figsize=(w_vedio/dpi_vedio, h_vedio/dpi_vedio), facecolor='white')
    plt.tight_layout()  # 尽量减少窗口的留白
    if show:
        plt.ion()  # 为了可以动态显示
    else:
        plt.ioff()

    for idx, item in enumerate(os.listdir(root_dataset)):
        log_BLUE('=' * 100)

        item_path = os.path.join(root_dataset, item)
        if not os.path.isdir(item_path):
            log_YELLOW('Skip {}, is not a folder'.format(item))
            continue

        name_group_folder = item

        # skip mixmode
        if 'mixmode' in name_group_folder:
            log_YELLOW('Skip {}, is mixmode'.format(name_group_folder))
            continue

        flag_skip = True
        for key in filter_keys:
            if key in name_group_folder:
                flag_skip = False
                break
        if flag_skip:
            log_YELLOW('Skip {}, filtered by key={}'.format(name_group_folder, key))
            continue

        path_group_folder = os.path.join(root_dataset, name_group_folder)

        names_frame_folder = os.listdir(path_group_folder)
        # sort, otherwise will occur frame_idx mess
        names_frame_folder.sort()
        num_frames = len(names_frame_folder)

        output_path = os.path.join(root_output, '{}.mp4'.format(name_group_folder))
        if not cover and os.path.exists(output_path):
            log_YELLOW('Skip {}, already generate {}'.format(name_group_folder, output_path))
            continue

        writer = FFMpegWriter(fps=10)
        with writer.saving(fig, output_path, dpi=dpi_vedio):
            for id_frame, name_frame_folder in enumerate(names_frame_folder):
                assert id_frame == int(name_frame_folder.replace('frame', ''))
                log('>>>> {} {}/{} {}'.format(name_group_folder, id_frame + 1, num_frames, name_frame_folder))

                # load data
                path_frame_folder = os.path.join(path_group_folder, name_frame_folder)
                frame = Frame(path_frame_folder)

                IRayCamera_json = frame.get_sensor_data('IRayCamera_json')
                IRayCamera_ts_str = IRayCamera_json['timestamp']
                IRayCamera_localtime = unix2local(IRayCamera_ts_str)
                IRayCamera_png = frame.get_sensor_data('IRayCamera_png')

                LeopardCamera0_json = frame.get_sensor_data('LeopardCamera0_json')
                LeopardCamera0_ts_str = LeopardCamera0_json['timestamp']
                LeopardCamera0_localtime = unix2local(LeopardCamera0_ts_str)
                LeopardCamera0_png = frame.get_sensor_data('LeopardCamera0_png')

                LeopardCamera1_json = frame.get_sensor_data('LeopardCamera1_json')
                LeopardCamera1_ts_str = LeopardCamera1_json['timestamp']
                LeopardCamera1_localtime = unix2local(LeopardCamera1_ts_str)
                LeopardCamera1_png = frame.get_sensor_data('LeopardCamera1_png')

                VelodyneLidar_json = frame.get_sensor_data('VelodyneLidar_json')
                VelodyneLidar_ts_str = VelodyneLidar_json['timestamp']
                VelodyneLidar_localtime = unix2local(VelodyneLidar_ts_str)
                VelodyneLidar_pcd = frame.get_sensor_data('VelodyneLidar_pcd')

                OCULiiRadar_json = frame.get_sensor_data('OCULiiRadar_json')
                OCULiiRadar_ts_str = OCULiiRadar_json['timestamp']
                OCULiiRadar_localtime = unix2local(OCULiiRadar_ts_str)
                OCULiiRadar_pcd = frame.get_sensor_data('OCULiiRadar_pcd')

                TIRadar_json = frame.get_sensor_data('TIRadar_json')
                TIRadar_ts_str = TIRadar_json['timestamp']
                TIRadar_localtime = unix2local(TIRadar_ts_str)
                TIRadar_pcd = frame.get_sensor_data('TIRadar_pcd')

                ts_offset_base = float(VelodyneLidar_ts_str)
                ts_offset_VelodyneLidar = float(VelodyneLidar_ts_str) - ts_offset_base
                ts_offset_OCULiiRadar = float(OCULiiRadar_ts_str) - ts_offset_base
                ts_offset_TIRadar = float(TIRadar_ts_str) - ts_offset_base

                '''
                visualize and plot
                row1: IRayCamera, LeopardCamera0, LeopardCamera1
                row2: OCULiiRadar, VelodyneLidar, TIRadar
                '''
                n_col_row1 = 3
                n_col_row2 = 3
                w_padding_row1 = 20
                w_padding_row2 = 200
                h_padding_up = 50
                h_padding_mid = 30
                h_padding_down = 0
                r = 40
                doppler_min = -5
                doppler_max = 5
                cmap = 'jet'
                lidar_intensity_min = 0
                lidar_intensity_max = 255
                colorbar_pad = 0.09

                IRayCamera_png_height, IRayCamera_png_width = IRayCamera_png.shape
                LeopardCamera0_png_height, LeopardCamera0_png_width, _ = LeopardCamera0_png.shape
                LeopardCamera1_png_height, LeopardCamera1_png_width, _ = LeopardCamera1_png.shape

                w_all = IRayCamera_png_width + LeopardCamera0_png_width + LeopardCamera1_png_width\
                        + (n_col_row1 + 1) * w_padding_row1

                pcd_size = (w_all - (n_col_row2 + 1) * w_padding_row2) / n_col_row2
                OCULiiRadar_pcd_height, OCULiiRadar_pcd_width = pcd_size, pcd_size
                VelodyneLidar_pcd_height, VelodyneLidar_pcd_width = pcd_size, pcd_size
                TIRadar_pcd_height, TIRadar_pcd_width = pcd_size, pcd_size

                h_row1 = np.array([IRayCamera_png_height, LeopardCamera0_png_height, LeopardCamera1_png_height]).max()
                h_row2 = np.array([OCULiiRadar_pcd_height, VelodyneLidar_pcd_height, TIRadar_pcd_height]).max()
                h_all = h_padding_up + h_row1 + h_padding_mid + h_row2 + h_padding_down

                # IRayCamera
                left1 = w_padding_row1 / w_all
                bottom1 = (h_padding_mid + h_row2 + h_padding_down) / h_all
                w1 = IRayCamera_png_width / w_all
                h1 = h_row1 / h_all
                ax1 = fig.add_axes([left1, bottom1, w1, h1])
                ax1.imshow(IRayCamera_png, cmap='gray')
                ax1.axis('off')
                ax1.set_title('IRayCamera({}/{})\n{}'.format(id_frame + 1, num_frames, IRayCamera_localtime))

                # LeopardCamera0
                left2 = left1 + w1 + w_padding_row1 / w_all
                bottom2 = (h_padding_mid + h_row2 + h_padding_down) / h_all
                w2 = LeopardCamera0_png_width / w_all
                h2 = h_row1 / h_all
                ax2 = fig.add_axes([left2, bottom2, w2, h2])
                ax2.imshow(LeopardCamera0_png)
                ax2.axis('off')
                ax2.set_title('LeopardCamera0({}/{})\n{}'.format(id_frame + 1, num_frames, LeopardCamera0_localtime))

                # LeopardCamera1
                left3 = left2 + w2 + w_padding_row1 / w_all
                bottom3 = (h_padding_mid + h_row2 + h_padding_down) / h_all
                w3 = LeopardCamera1_png_width / w_all
                h3 = h_row1 / h_all
                ax3 = fig.add_axes([left3, bottom3, w3, h3])
                ax3.imshow(LeopardCamera1_png)
                ax3.axis('off')
                ax3.set_title('LeopardCamera1({}/{})\n{}'.format(id_frame + 1, num_frames, LeopardCamera1_localtime))

                # OCULiiRadar
                left5 = w_padding_row2 / w_all
                bottom5 = h_padding_down / h_all
                w5 = OCULiiRadar_pcd_width / w_all
                h5 = h_row2 / h_all
                ax5 = fig.add_axes([left5, bottom5, w5, h5])
                OCULiiRadar_pcd_in_zone = pcd_in_zone(OCULiiRadar_pcd, xlim=[-r, r], zlim=[0, 2 * r])
                ax5.scatter(
                    OCULiiRadar_pcd_in_zone[:, 0], OCULiiRadar_pcd_in_zone[:, 2],
                    s=np.round(OCULiiRadar_pcd_in_zone[:, 4]),
                    c=np.round(OCULiiRadar_pcd_in_zone[:, 3]),
                    cmap=cmap, vmin=doppler_min, vmax=doppler_max
                )
                ax5.set_xlim([-r, r])
                ax5.set_ylim([0, 2 * r])
                ax5.set_xlabel('x/m')
                ax5.set_ylabel('z/m')
                ax5.grid(color='gray', linestyle='-', linewidth=1)
                ax5.set_aspect(1)
                ax5.set_title('OCULiiRadar({}/{})\n{}'.format(id_frame + 1, num_frames, OCULiiRadar_localtime))
                plt.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=doppler_min, vmax=doppler_max)),
                    ax=ax5, location='bottom', orientation='horizontal', label='doppler(m/s)', pad=colorbar_pad
                )
                if ts_offset_OCULiiRadar >= 0:
                    ax5.text(
                        -r, 2 * r, '+{:.0f} ms'.format(np.abs(ts_offset_OCULiiRadar*1000)),
                        {'color': 'red', 'fontsize': 14, 'fontweight': 'bold'}
                    )
                else:
                    ax5.text(
                        -r, 2 * r, '-{:.0f} ms'.format(np.abs(ts_offset_OCULiiRadar * 1000)),
                        {'color': 'green', 'fontsize': 14, 'fontweight': 'bold'}
                    )

                # VelodyneLidar
                left6 = left5 + w5 + w_padding_row2 / w_all
                bottom6 = h_padding_down / h_all
                w6 = VelodyneLidar_pcd_width / w_all
                h6 = h_row2 / h_all
                ax6 = fig.add_axes([left6, bottom6, w6, h6])
                VelodyneLidar_pcd_in_zone = pcd_in_zone(VelodyneLidar_pcd, xlim=[-r, r], ylim=[0, 2 * r])
                ax6.scatter(
                    VelodyneLidar_pcd_in_zone[:, 0], VelodyneLidar_pcd_in_zone[:, 1],
                    s=1,
                    c=np.round(VelodyneLidar_pcd_in_zone[:, 3]),
                    cmap=cmap, vmin=lidar_intensity_min, vmax=lidar_intensity_max
                )
                ax6.set_xlim([-r, r])
                ax6.set_ylim([0, 2 * r])
                ax6.set_xlabel('x/m')
                ax6.set_ylabel('y/m')
                ax6.grid(color='gray', linestyle='-', linewidth=1)
                ax6.set_aspect(1)
                ax6.set_title('VelodyneLidar({}/{})\n{}'.format(id_frame + 1, num_frames,VelodyneLidar_localtime))
                plt.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lidar_intensity_min, vmax=lidar_intensity_max)),
                    ax=ax6, location='bottom', orientation='horizontal', label='intensity', pad=colorbar_pad
                )
                if ts_offset_VelodyneLidar >= 0:
                    ax6.text(
                        -r, 2 * r, '+{:.0f} ms'.format(np.abs(ts_offset_VelodyneLidar*1000)),
                        {'color': 'red', 'fontsize': 14, 'fontweight': 'bold'}
                    )
                else:
                    ax6.text(
                        -r, 2 * r, '-{:.0f} ms'.format(np.abs(ts_offset_VelodyneLidar * 1000)),
                        {'color': 'green', 'fontsize': 14, 'fontweight': 'bold'}
                    )

                # TIRadar
                left7 = left6 + w6 + w_padding_row2 / w_all
                bottom7 = h_padding_down / h_all
                w7 = TIRadar_pcd_width / w_all
                h7 = h_row2 / h_all
                ax7 = fig.add_axes([left7, bottom7, w7, h7])
                TIRadar_pcd_in_zone = pcd_in_zone(TIRadar_pcd, xlim=[-r, r], ylim=[0, 2 * r])
                ax7.scatter(
                    TIRadar_pcd_in_zone[:, 0], TIRadar_pcd_in_zone[:, 1],
                    s=np.round(TIRadar_pcd_in_zone[:, 4]),
                    c=np.round(TIRadar_pcd_in_zone[:, 3]),
                    cmap=cmap, vmin=doppler_min, vmax=doppler_max
                )
                ax7.set_xlim([-r, r])
                ax7.set_ylim([0, 2 * r])
                ax7.set_xlabel('x/m')
                ax7.set_ylabel('y/m')
                ax7.grid(color='gray', linestyle='-', linewidth=1)
                ax7.set_aspect(1)
                ax7.set_title('TIRadar({}/{})\n{}'.format(id_frame + 1, num_frames, TIRadar_localtime))
                plt.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=doppler_min, vmax=doppler_max)),
                    ax=ax7, location='bottom', orientation='horizontal', label='doppler(m/s)', pad=colorbar_pad
                )
                if ts_offset_TIRadar >= 0:
                    ax7.text(
                        -r, 2 * r, '+{:.0f} ms'.format(np.abs(ts_offset_TIRadar*1000)),
                        {'color': 'red', 'fontsize': 14, 'fontweight': 'bold'}
                    )
                else:
                    ax7.text(
                        -r, 2 * r, '-{:.0f} ms'.format(np.abs(ts_offset_TIRadar * 1000)),
                        {'color': 'green', 'fontsize': 14, 'fontweight': 'bold'}
                    )

                writer.grab_frame()
                if show:
                    plt.pause(0.1)
                # plt.show()
                plt.clf()

        log_GREEN('Generate {}'.format(output_path))


if __name__ == '__main__':
    main()

