import os
import sys
import glob
import argparse

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import pandas as pd
from dataset_v2 import load_json, load_TIRadar_adcdata, load_TIRadar_calibmat, unix2local
from utils import *
from TIRadar.signal_process import NormalModeProcess

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dataset', type=str, default='/mnt/Dataset/ourDataset_v2', help='ourDataset_v2 path')
    parser.add_argument('--generate_pcd', type=bool, default=True, help='flag whether generate pcd')
    parser.add_argument('--generate_heatmapBEV', type=bool, default=True, help='flag whether generate heatmapBEV')
    parser.add_argument('--generate_heatmap4D', type=bool, default=False, help='flag whether generate heatmap4D')

    parser.add_argument('--filter_keys', type=str, nargs='*', default=[], help='keys for filter')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    root_dataset = args.root_dataset
    generate_pcd = args.generate_pcd
    generate_heatmapBEV = args.generate_heatmapBEV
    generate_heatmap4D = args.generate_heatmap4D
    filter_keys = args.filter_keys

    for idx, item in enumerate(os.listdir(root_dataset)):

        item_path = os.path.join(root_dataset, item)
        if not os.path.isdir(item_path):
            log_YELLOW('Skip {}, is not a folder'.format(item))
            continue

        name_group_folder = item

        # skip mixmode
        if 'mixmode' in name_group_folder:
            log_YELLOW('Skip {}, is mixmode'.format(name_group_folder))
            continue

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
        num_frames = len(names_frame_folder)

        for id_frame, name_frame_folder in enumerate(names_frame_folder):
            log_BLUE('='*100)
            path_frame_folder = os.path.join(path_group_folder, name_frame_folder)
            path_TIRadar_folder = os.path.join(path_frame_folder, 'TIRadar')

            flag_skip = True
            if generate_pcd and len(glob.glob(os.path.join(path_TIRadar_folder, '*.pcd'))) == 0:
                flag_skip = False
            if generate_heatmapBEV and len(glob.glob(os.path.join(path_TIRadar_folder, '*.heatmapBEV.npz'))) == 0:
                flag_skip = False
            if generate_heatmap4D and len(glob.glob(os.path.join(path_TIRadar_folder, '*.heatmap4D.npz'))) == 0:
                flag_skip = False
            if flag_skip:
                log_YELLOW('Skip, results have already been generated')
                continue

            TIRadar_json = load_json(glob.glob(os.path.join(path_TIRadar_folder, '*.json'))[0])
            timestamp_unix = TIRadar_json['timestamp']
            localtime = unix2local(timestamp_unix)
            TIRadar_npz = load_TIRadar_adcdata(glob.glob(os.path.join(path_TIRadar_folder, '*.npz'))[0])
            TIRadar_calibmat = load_TIRadar_calibmat(glob.glob(os.path.join(path_TIRadar_folder, '*.mat'))[0])

            nmp = NormalModeProcess(TIRadar_npz['mode_infos'], TIRadar_npz['data_real'], TIRadar_npz['data_imag'], TIRadar_calibmat)
            nmp.run(generate_pcd=generate_pcd, generate_heatmapBEV=generate_heatmapBEV, generate_heatmap4D=generate_heatmap4D)

            pcd = nmp.get_pcd()
            if generate_pcd:
                pcd_path = os.path.join(path_TIRadar_folder, '{}.pcd'.format(timestamp_unix))

                if pcd is not None:
                    df_pcd = pd.DataFrame({
                        'x': pcd['x'].astype('float32'),
                        'y': pcd['y'].astype('float32'),
                        'z': pcd['z'].astype('float32'),
                        'doppler': pcd['doppler'].astype('float32'),
                        'snr': pcd['snr'].astype('float32'),
                        'intensity': pcd['intensity'].astype('float32'),
                        'noise': pcd['noise'].astype('float32')
                    })
                else:
                    df_pcd = pd.DataFrame(columns=['x', 'y', 'z', 'doppler', 'snr', 'intensity', 'noise'])

                df_pcd.to_csv(pcd_path, sep=' ', index=False, header=False)
                with open(pcd_path, 'r') as f_pcd:
                    lines = f_pcd.readlines()
                with open(pcd_path, 'w') as f_pcd:
                    f_pcd.write('VERSION .7\n')
                    f_pcd.write('FIELDS')
                    for col in df_pcd.columns.values:
                        f_pcd.write(' {}'.format(col))
                    f_pcd.write('\n')
                    f_pcd.write('SIZE 4 4 4 4 4 4\n')
                    f_pcd.write('TYPE F F F F F F\n')
                    f_pcd.write('COUNT 1 1 1 1 1 1\n')
                    f_pcd.write('WIDTH {}\n'.format(len(pcd)))
                    f_pcd.write('HEIGHT 1\n')
                    f_pcd.write('VIEWPOINT 0 0 0 1 0 0 0\n')
                    f_pcd.write('POINTS {}\n'.format(len(pcd)))
                    f_pcd.write('DATA ascii\n')
                    f_pcd.writelines(lines)
                log_GREEN('Save {}'.format(os.path.basename(pcd_path)))

            heatmapBEV = nmp.get_heatmapBEV()
            if generate_heatmapBEV and heatmapBEV is not None:
                heatmapBEV_path = os.path.join(path_TIRadar_folder, '{}.heatmapBEV.npz'.format(timestamp_unix))
                np.savez(
                    heatmapBEV_path,
                    x=heatmapBEV['x'],
                    y=heatmapBEV['y'],
                    heatmapBEV_static=heatmapBEV['heatmapBEV_static'],
                    heatmapBEV_dynamic=heatmapBEV['heatmapBEV_dynamic']
                )
                log_GREEN('Save {}'.format(os.path.basename(heatmapBEV_path)))

            log_GREEN('{} | Frame({}/{}) | Timestamp_unix: {} | Localtime: {}, complete.'.format(
                name_group_folder, id_frame+1, num_frames, timestamp_unix, localtime
            ))

if __name__ == '__main__':
    main()
