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


def main():
    root_dataset_to_label = '/mnt/Dataset/ourDataset_v2_label'
    root_output = '/mnt/Dataset/data_to_label'
    camera_list = ['LeopardCamera0']
    pcd_sensor = 'VelodyneLidar'
    r = 100
    xlim = [-r, r]
    ylim = [0, r]
    num_group_per_package = 20

    check_to_create_path(root_output)

    groupnames = os.listdir(root_dataset_to_label)
    groupnames.sort()
    num_group = len(groupnames)

    for idx_group, groupname in enumerate(groupnames):
        idx_package = idx_group // num_group_per_package
        root_output_package = os.path.join(root_output, 'package{}'.format(idx_package+1))
        check_to_create_path(root_output_package)

        log('='*100)
        log('Package {} - Process {}/{} {}'.format(idx_package+1, idx_group+1, num_group, groupname))

        group_path = os.path.join(root_dataset_to_label, groupname)
        framenames = os.listdir(group_path)
        framenames.sort()
        num_frame = len(framenames)

        root_output_group = os.path.join(root_output_package, groupname)
        check_to_create_path(root_output_group)

        root_output_group_calib = os.path.join(root_output_group, 'calib')
        check_to_create_path(root_output_group_calib)

        root_output_group_image = os.path.join(root_output_group, 'image')
        check_to_create_path(root_output_group_image)

        for camera in camera_list:
            check_to_create_path(os.path.join(root_output_group_image, camera))

        root_output_group_pcd = os.path.join(root_output_group, 'pcd')
        check_to_create_path(root_output_group_pcd)

        root_output_group_label = os.path.join(root_output_group, 'label')
        check_to_create_path(root_output_group_label)

        for idx_frame, framename in enumerate(framenames):
            log('=' * 100)
            log('Process {}/{} {}, {}/{} {}'.format(idx_frame + 1, num_frame, framename, idx_group+1, num_group, groupname))

            frame_path = os.path.join(group_path, framename)

            # create .json in calib
            if idx_frame == 0:
                for camera in camera_list:
                    camera_json_path = glob.glob(os.path.join(frame_path, camera, '*.json'))[0]
                    camera_json = load_json(camera_json_path)
                    intrinsic = camera_json['intrinsic_matrix']
                    intrinsic = np.array(intrinsic).reshape((-1)).tolist()

                    pcd_sensor_json_path = glob.glob(os.path.join(frame_path, pcd_sensor, '*.json'))[0]
                    pcd_sensor_json = load_json(pcd_sensor_json_path)

                    extrinsic_name = '{}_to_{}_extrinsic'.format(pcd_sensor, camera)
                    if extrinsic_name in pcd_sensor_json.keys():
                        extrinsic = pcd_sensor_json[extrinsic_name]
                        extrinsic = np.array(extrinsic).reshape((-1)).tolist()
                    else:
                        pcd_sensor_to_LeopardCamera0_extrinsic = pcd_sensor_json['{}_to_LeopardCamera0_extrinsic'.format(pcd_sensor)]
                        TIRadar_json_path = glob.glob(os.path.join(frame_path, 'TIRadar', '*.json'))[0]
                        TIRadar_json = load_json(TIRadar_json_path)
                        TIRadar_to_LeopardCamera0_extrinsic = TIRadar_json['TIRadar_to_LeopardCamera0_extrinsic']
                        TIRadar_to_camera_extrinsic = TIRadar_json['TIRadar_to_{}_extrinsic'.format(camera)]
                        LeopardCamera0_to_camera_extrinsic = np.matmul(
                            np.array(TIRadar_to_camera_extrinsic),
                            np.linalg.inv(np.array(TIRadar_to_LeopardCamera0_extrinsic))
                        )
                        extrinsic = np.matmul(LeopardCamera0_to_camera_extrinsic, pcd_sensor_to_LeopardCamera0_extrinsic)
                        extrinsic = extrinsic.reshape((-1)).tolist()

                    calib_dst_path = os.path.join(root_output_group_calib, '{}.json'.format(camera))
                    save_dict_as_json(
                        calib_dst_path,
                        {
                            'intrinsic': intrinsic,
                            'extrinsic': extrinsic
                        }
                    )

            # link image
            for camera in camera_list:
                image_src_path = glob.glob(os.path.join(frame_path, camera, '*.png'))[0]
                image_dst_path = os.path.join(root_output_group_image, camera, '{}.png'.format(framename))

                try:
                    os.remove(image_dst_path)
                    log_YELLOW('Remove {}'.format(image_dst_path))
                except:
                    pass
                os.symlink(image_src_path, image_dst_path)
                log_GREEN('Link to {}'.format(image_dst_path))

            # crop and save pcd
            pcd_src_path = glob.glob(os.path.join(frame_path, pcd_sensor, '*.pcd'))[0]
            pcd_dst_path = os.path.join(root_output_group_pcd, '{}.pcd'.format(framename))

            pcd = load_pcd(pcd_src_path, pcd_sensor)
            pcd_cropped = pcd_in_zone(pcd, xlim=xlim, ylim=ylim, return_type='dict')
            save_dict_as_VelodyneLidar_pcd(pcd_dst_path, pcd_cropped)

            log_GREEN('Crop and Save {}'.format(pcd_dst_path))

        # whether generate zip file or not
        if np.mod((idx_group+1), num_group_per_package) == 0:
            zip_path = root_output_package + '.zip'
            z = zipfile.ZipFile(zip_path, 'w')
            for path, dirnames, filenames in os.walk(root_output_package):
                fpath = path.replace(root_output_package, '')
                if fpath != '':
                    z.write(path, fpath)
                for filename in filenames:
                    z.write(os.path.join(path, filename), os.path.join(fpath, filename))
            z.close()

            log_GREEN('Compress and save to {}'.format(zip_path))



if __name__ == '__main__':
    main()
