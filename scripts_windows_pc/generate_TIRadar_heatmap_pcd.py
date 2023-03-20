import glob
import os
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import pandas as pd
from dataset_v2 import load_json, load_TIRadar_npz, load_TIRadar_calibmat, unix2local
from utils import *
from TIRadar.signal_process import NormalModeProcess

def main():
    root_dataset = 'F:\\ourDataset_v2'
    names_group_folder = os.listdir(root_dataset)
    num_groups = len(names_group_folder)
    generate_pcd = True
    generate_heatmapBEV = True
    generate_heatmap4D = False

    for idx_group, name_group_folder in enumerate(names_group_folder):
        path_group_folder = os.path.join(root_dataset, name_group_folder)
        names_frame_folder = os.listdir(path_group_folder)
        num_frames = len(names_frame_folder)

        for id_frame, name_frame_folder in enumerate(names_frame_folder):
            log_BLUE('='*100)
            path_frame_folder = os.path.join(path_group_folder, name_frame_folder)
            path_TIRadar_folder = os.path.join(path_frame_folder, 'TIRadar')

            TIRadar_json = load_json(glob.glob(os.path.join(path_TIRadar_folder, '*.json'))[0])
            timestamp_unix = TIRadar_json['timestamp']
            localtime = unix2local(timestamp_unix)
            TIRadar_npz = load_TIRadar_npz(glob.glob(os.path.join(path_TIRadar_folder, '*.npz'))[0])
            TIRadar_calibmat = load_TIRadar_calibmat(glob.glob(os.path.join(path_TIRadar_folder, '*.mat'))[0])

            nmp = NormalModeProcess(TIRadar_npz['mode_infos'], TIRadar_npz['data_real'], TIRadar_npz['data_imag'], TIRadar_calibmat)
            nmp.run(generate_pcd=generate_pcd, generate_heatmapBEV=generate_heatmapBEV, generate_heatmap4D=generate_heatmap4D)

            pcd = nmp.get_pcd()
            if generate_pcd and pcd is not None:
                pcd_path = os.path.join(path_TIRadar_folder, '{}.pcd'.format(timestamp_unix))

                df_pcd = pd.DataFrame({
                    'x': pcd['x'].values.astype('float32'),
                    'y': pcd['y'].values.astype('float32'),
                    'z': pcd['z'].values.astype('float32'),
                    'doppler': pcd['doppler'].values.astype('float32'),
                    'snr': pcd['snr'].values.astype('float32'),
                    'intensity': pcd['intensity'].values.astype('float32')
                })
                df_pcd.to_csv(pcd_path, sep=' ', index=False, header=False)
                with open(pcd_path, 'r') as f_pcd:
                    lines = f_pcd.readlines()
                with open(pcd_path, 'w') as f_pcd:
                    f_pcd.write('VERSION .7\n')
                    f_pcd.write('FIELDS')
                    for col in pcd.columns.values:
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
                log('Save {}'.format(os.path.basename(pcd_path)))

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
                log('Save {}'.format(os.path.basename(heatmapBEV_path)))

            log_GREEN('Group({}/{}) | Frame({}/{}) | Timestamp_unix: {} | Localtime: {}, complete.'.format(
                idx_group+1, num_groups, id_frame+1, num_frames, timestamp_unix, localtime
            ))

if __name__ == '__main__':
    main()
