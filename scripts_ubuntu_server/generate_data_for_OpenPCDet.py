import os
import sys
import glob

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np

from dataset_v2 import log, log_GREEN, load_json, save_dict_as_json, log_YELLOW, load_pcd, pcd_in_zone, save_dict_as_VelodyneLidar_pcd

def check_to_create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        log_GREEN('Create {}'.format(path))

def main():
    root_dataset = '/mnt/Dataset/ourDataset_v2'
    filter_keys = ['20221217_group0012_mode1_369frames']
    root_output = '/mnt/Dataset/data_for_OpenPCDet'
    pcd_sensor = 'VelodyneLidar'
    r = 100
    xlim = [-r, r]
    ylim = [0, r]

    check_to_create_path(root_output)

    groupnames = os.listdir(root_dataset)
    groupnames.sort()
    num_group = len(groupnames)

    for idx_group, groupname in enumerate(groupnames):
        log('='*100)

        if len(filter_keys) > 0:
            flag_skip = True
            for key in filter_keys:
                if key in groupname:
                    flag_skip = False
                    break
            if flag_skip:
                log_YELLOW('Skip {}, filtered by key={}'.format(groupname, filter_keys))
                continue

        log('Process {}/{} {}'.format(idx_group+1, num_group, groupname))

        group_path = os.path.join(root_dataset, groupname)
        framenames = os.listdir(group_path)
        framenames.sort()
        num_frame = len(framenames)

        root_output_group = os.path.join(root_output, groupname)
        check_to_create_path(root_output_group)

        root_output_group_pcd = os.path.join(root_output_group, 'pcd')
        check_to_create_path(root_output_group_pcd)

        for idx_frame, framename in enumerate(framenames):
            log('=' * 100)
            log('Process {}/{} {}, {}/{} {}'.format(idx_frame + 1, num_frame, framename, idx_group+1, num_group, groupname))

            frame_path = os.path.join(group_path, framename)

            pcd_src_path = glob.glob(os.path.join(frame_path, pcd_sensor, '*.pcd'))[0]
            pcd_dst_path = os.path.join(root_output_group_pcd, '{}.npy'.format(framename))

            pcd = load_pcd(pcd_src_path, pcd_sensor)
            # Transform your point cloud data
            '''
                You need to transform the coordinate of your custom point cloud to the unified normative coordinate
                of OpenPCDet, that is, x-axis points towards to front direction, y-axis points towards to the left 
                direction, and z-axis points towards to the top direction.
                
                (Optional) the z-axis origin of your point cloud coordinate should be about 1.6m above the ground 
                surface, since currently the provided models are trained on the KITTI dataset.
                
                Set the intensity information, and save your transformed custom data to numpy file
            '''
            # Save it to the file.
            # The shape of points should be (num_points, 4), that is [x, y, z, intensity] (Only for KITTI dataset).
            # If you doesn't have the intensity information, just set them to zeros.
            # If you have the intensity information, you should normalize them to [0, 1].
            pcd_cropped = pcd_in_zone(pcd, xlim=xlim, ylim=ylim)

            points = np.zeros((pcd_cropped.shape[0], 4), dtype='float')
            points[:, 0] = pcd_cropped[:, 1]
            points[:, 1] = -pcd_cropped[:, 0]
            points[:, 2] = pcd_cropped[:, 2]
            points[:, 3] = pcd_cropped[:, 3] / 255

            np.save(pcd_dst_path, points)

            log_GREEN('Crop and Save {}'.format(pcd_dst_path))


if __name__ == '__main__':
    main()
