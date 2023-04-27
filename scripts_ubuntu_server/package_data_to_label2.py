import os
import sys
import glob
import zipfile

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np

from dataset_v2 import log, log_GREEN, load_json, save_dict_as_json, log_YELLOW, load_pcd, pcd_in_zone, save_dict_as_VelodyneLidar_pcd

def check_to_create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        log_GREEN('Create {}'.format(path))

def write_in_zip(z, path, bash_path):
    fpath = path.replace(bash_path, '')
    z.write(path, fpath)

def main():
    root_data_for_SUSTechPOINTS = '/mnt/Dataset/data_for_SUSTechPOINTS'
    root_output = '/mnt/Dataset/data_to_label'
    filter_keys = ['IRayCamera', 'LeopardCamera1']
    num_group_per_package = 10

    check_to_create_path(root_output)

    groupnames = os.listdir(root_data_for_SUSTechPOINTS)
    groupnames.sort()
    num_group = len(groupnames)

    # select group in packages
    packages = []
    idx_group = 0
    while True:
        groups_in_package = []
        for i in range(num_group_per_package):
            groups_in_package.append(groupnames[idx_group])
            idx_group += 1
            if idx_group >= num_group:
                break

        packages.append(groups_in_package)
        if idx_group >= num_group:
            break

    # generate .zip file
    cnt_groups = 0
    for idx_package in range(len(packages)):
        zip_path = os.path.join(root_output, 'package{}.zip'.format(idx_package + 1))
        z = zipfile.ZipFile(zip_path, 'w')

        groups_in_package = packages[idx_package]
        for idx_group_in_package, groupname in enumerate(groups_in_package):
            cnt_groups += 1
            log('{}/{} | {}/{} {}'.format(cnt_groups, num_group, idx_group_in_package+1, len(groups_in_package), groupname))

            root_group = os.path.join(root_data_for_SUSTechPOINTS, groupname)
            write_in_zip(z, root_group, root_data_for_SUSTechPOINTS)

            root_group_calib = os.path.join(root_group, 'calib')
            write_in_zip(z, root_group_calib, root_data_for_SUSTechPOINTS)
            items = os.listdir(root_group_calib)
            items.sort()
            for item in items:
                save_flag = True
                for key in filter_keys:
                    if key in item:
                        save_flag = False
                        break
                if save_flag:
                    item_path = os.path.join(root_group_calib, item)
                    write_in_zip(z, item_path, root_data_for_SUSTechPOINTS)

            root_group_image = os.path.join(root_group, 'image')
            write_in_zip(z, root_group_image, root_data_for_SUSTechPOINTS)
            subfolders = os.listdir(root_group_image)
            subfolders.sort()
            for subfolder in subfolders:
                save_flag = True
                for key in filter_keys:
                    if key in subfolder:
                        save_flag = False
                        break
                if save_flag:
                    subfolder_path = os.path.join(root_group_image, subfolder)
                    write_in_zip(z, subfolder_path, root_data_for_SUSTechPOINTS)
                    items = os.listdir(subfolder_path)
                    items.sort()
                    for item in items:
                        item_path = os.path.join(subfolder_path, item)
                        write_in_zip(z, item_path, root_data_for_SUSTechPOINTS)


            root_group_label = os.path.join(root_group, 'label')
            write_in_zip(z, root_group_label, root_data_for_SUSTechPOINTS)

            root_group_pcd = os.path.join(root_group, 'pcd')
            write_in_zip(z, root_group_pcd, root_data_for_SUSTechPOINTS)
            items = os.listdir(root_group_pcd)
            items.sort()
            for item in items:
                item_path = os.path.join(root_group_pcd, item)
                write_in_zip(z, item_path, root_data_for_SUSTechPOINTS)

        z.close()
        log_GREEN('({}/{}) Compress and save to {}'.format(idx_package+1, len(packages), zip_path))




if __name__ == '__main__':
    main()
