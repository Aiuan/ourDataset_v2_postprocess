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
    parser.add_argument('--root_output', type=str, default='/mnt/Dataset/ourDataset_v2_vis/vis_3cameras', help='ourDataset_v2 path')
    parser.add_argument('--show', type=bool, default=False, help='dynamic show or not')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    root_dataset = args.root_dataset
    filter_keys = args.filter_keys
    root_output = args.root_output
    show = args.show

    if not os.path.exists(root_output):
        os.makedirs(root_output)
        log_GREEN('create {}'.format(root_output))

    fig = plt.figure(figsize=(19.2, 5.4), facecolor='white')
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
        flag_skip = False
        for key in filter_keys:
            if not (key in name_group_folder):
                log_YELLOW('Skip {}, filtered by key={}'.format(name_group_folder, key))
                flag_skip = True
                break
        if flag_skip:
            continue

        path_group_folder = os.path.join(root_dataset, name_group_folder)
        names_frame_folder = os.listdir(path_group_folder)
        # sort, otherwise will occur frame_idx mess
        names_frame_folder.sort()
        num_frames = len(names_frame_folder)

        output_path = os.path.join(root_output, '{}.mp4'.format(name_group_folder))
        writer = FFMpegWriter(fps=10)
        with writer.saving(fig, output_path, dpi=100):
            for id_frame, name_frame_folder in enumerate(names_frame_folder):
                assert id_frame == int(name_frame_folder.replace('frame', ''))
                log('>>>> {} {}/{} frame'.format(name_group_folder, id_frame + 1, num_frames))

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

                IRayCamera_png_height, IRayCamera_png_width = IRayCamera_png.shape
                LeopardCamera0_png_height, LeopardCamera0_png_width, _ = LeopardCamera0_png.shape
                LeopardCamera1_png_height, LeopardCamera1_png_width, _ = LeopardCamera1_png.shape

                n_col = 3
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
                h1 = 0.85

                ax1 = fig.add_axes((w0, (1-h1)/3, w1, h1))
                ax1.imshow(IRayCamera_png, cmap='gray')
                ax1.axis('off')
                ax1.set_title('IRayCamera({}/{})\n{}'.format(id_frame + 1, num_frames, IRayCamera_localtime))

                ax2 = fig.add_axes((w0 + w1 + w0, (1-h1)/3, w2, h1))
                ax2.imshow(LeopardCamera0_png)
                ax2.axis('off')
                ax2.set_title('LeopardCamera0({}/{})\n{}'.format(id_frame + 1, num_frames, LeopardCamera0_localtime))

                ax3 = fig.add_axes((w0 + w1 + w0 + w2 + w0, (1-h1)/3, w3, h1))
                ax3.imshow(LeopardCamera1_png)
                ax3.axis('off')
                ax3.set_title('LeopardCamera1({}/{})\n{}'.format(id_frame + 1, num_frames, LeopardCamera1_localtime))

                writer.grab_frame()
                if show:
                    plt.pause(0.1)
                plt.clf()

        log_GREEN('Generate {}'.format(output_path))

if __name__ == '__main__':
    main()

