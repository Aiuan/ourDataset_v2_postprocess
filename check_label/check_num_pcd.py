import os
import sys
import glob
import argparse

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset_v2 import log, log_GREEN, load_json, load_VelodyneLidar_pcd

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--root_dataset', type=str, default='/mnt/Dataset/ourDataset_v2_label')
    parser.add_argument('--root_dataset', type=str, default='/mnt/Dataset/data_for_SUSTechPOINTS')

    parser.add_argument('--root_label', type=str, default='/mnt/Dataset/labels_for_checking2')

    parser.add_argument('--label_foldername', type=str, default='', help='label folder name')

    parser.add_argument('--thred', type=int, default=200)

    parser.add_argument('--output_foldername', type=str, default='runs2/num_pcd_less200', help='output folder name')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    root_dataset = args.root_dataset
    root_label = args.root_label
    label_foldername = args.label_foldername
    thred = args.thred
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
        df_warning = pd.DataFrame()

        label_folder = os.path.join(root_label, label_foldername)
        groupnames = os.listdir(label_folder)
        groupnames.sort()
        for i, groupname in enumerate(groupnames):
            log('=' * 100)
            log('{}/{} {}'.format(i + 1, len(groupnames), groupname))

            group_folder = os.path.join(label_folder, groupname)
            framenames = os.listdir(group_folder)
            framenames = [framename.split('.')[0] for framename in framenames]
            framenames.sort()
            for j, framename in enumerate(framenames):
                log('{}/{} {}'.format(j + 1, len(framenames), framename))

                labels = load_json(os.path.join(group_folder, '{}.json'.format(framename)))

                # pcd_path = glob.glob(os.path.join(root_dataset, groupname, framename, 'VelodyneLidar', '*.pcd'))[0]
                pcd_path = os.path.join(root_dataset, groupname, 'pcd', '{}.pcd'.format(framename))
                pcd = load_VelodyneLidar_pcd(pcd_path)
                xyz = np.stack((pcd['x'], pcd['y'], pcd['z']))

                # plt.figure()
                # plt.scatter(xyz[0, :], xyz[1, :], 1)

                for k, label in enumerate(labels):
                    log('{}/{} object_id: {}'.format(k+1, len(labels), label['object_id']))

                    pos_offset = np.array([label['x'], label['y'], label['z']]).reshape((-1, 1))

                    # plt.figure()
                    # plt.scatter(xyz[0, :]-pos_offset[0, 0], xyz[1, :]-pos_offset[1, 0], 1)

                    alpha = label['alpha']
                    rot = np.array([
                        [np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 1]
                    ])
                    pcd_bbox = np.matmul(rot, xyz - pos_offset)

                    # plt.figure()
                    # plt.scatter(pcd_bbox[0, :], pcd_bbox[1, :], 1)

                    mask_x = np.logical_and(
                        pcd_bbox[0, :] >= -label['l']/2,
                        pcd_bbox[0, :] <= label['l']/2
                    )

                    mask_y = np.logical_and(
                        pcd_bbox[1, :] >= -label['w']/2,
                        pcd_bbox[1, :] <= label['w']/2
                    )

                    mask_z = np.logical_and(
                        pcd_bbox[2, :] >= -label['h']/2,
                        pcd_bbox[2, :] <= label['h']/2
                    )

                    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

                    pcd_bbox = pcd_bbox[:, mask]

                    # plt.figure()
                    # plt.scatter(pcd_bbox[0, :], pcd_bbox[1, :], 1)
                    # plt.show()

                    n = pcd_bbox.shape[1]

                    if n <= thred:
                        df_tmp = pd.DataFrame([label])
                        df_tmp['num_pcd'] = n
                        df_tmp['framename'] = framename
                        df_tmp['groupname'] = groupname
                        df_warning = pd.concat((df_warning, df_tmp), ignore_index=True)

        df_warning.to_excel(os.path.join(root_output, '{}-numPcdWarning.xlsx'.format(label_foldername)))


    print('done')


if __name__ == '__main__':
    main()

