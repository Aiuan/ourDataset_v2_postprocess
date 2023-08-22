import os
import glob
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np

from dataset_v2 import log, log_GREEN, load_json, save_dict_as_json

def main():
    root_data = '/mnt/Dataset/ourDataset_v2'
    root_calib = '/mnt/Dataset/sensors_calibration_v2.1/results'
    map = {
        '20221217': 'zhoushan',
        '20221219': 'zhoushan',
        '20221220': 'zhoushan',
        '20221221': 'zhoushan',
        '20221223': 'yantai',
        '20221224': 'yantai'
    }

    groupnames = os.listdir(root_data)
    groupnames.sort()
    for i, groupname in enumerate(groupnames):
        log('='*100)
        log('{}/{} {}'.format(i+1, len(groupnames), groupname))

        group_folder = os.path.join(root_data, groupname)
        framenames = os.listdir(group_folder)
        framenames.sort()

        exp_name = groupname.split('_')[0]
        calibres_name = map[exp_name]
        calibres_folder = os.path.join(root_calib, calibres_name)

        for j, framename in enumerate(framenames):

            frame_folder = os.path.join(group_folder, framename)

            # IRayCamera
            json_path = glob.glob(os.path.join(frame_folder, 'IRayCamera', '*.json'))[0]
            json_old = load_json(json_path)
            intrinsic = load_json(os.path.join(calibres_folder, 'IRayCamera_intrinsic.json'))
            extrinsic = load_json(os.path.join(calibres_folder, 'IRayCamera_to_LeopardCamera0_extrinsic.json'))
            json_new = {
                'image_size': intrinsic['image_size'],
                'intrinsic': intrinsic['intrinsic_matrix'],
                'radial_distortion': intrinsic['radial_distortion'],
                'tangential_distortion': intrinsic['tangential_distortion'],
                'timestamp': json_old['timestamp'],
                'IRayCamera_to_LeopardCamera0_extrinsic': extrinsic['extrinsic_matrix']
            }
            save_dict_as_json(json_path, json_new)

            # LeopardCamera0
            json_path = glob.glob(os.path.join(frame_folder, 'LeopardCamera0', '*.json'))[0]
            json_old = load_json(json_path)
            intrinsic = load_json(os.path.join(calibres_folder, 'LeopardCamera0_intrinsic.json'))
            json_new = {
                'image_size': intrinsic['image_size'],
                'intrinsic': intrinsic['intrinsic_matrix'],
                'radial_distortion': intrinsic['radial_distortion'],
                'tangential_distortion': intrinsic['tangential_distortion'],
                'timestamp': json_old['timestamp']
            }
            save_dict_as_json(json_path, json_new)

            # LeopardCamera1
            json_path = glob.glob(os.path.join(frame_folder, 'LeopardCamera1', '*.json'))[0]
            json_old = load_json(json_path)
            intrinsic = load_json(os.path.join(calibres_folder, 'LeopardCamera1_intrinsic.json'))
            extrinsic = load_json(os.path.join(calibres_folder, 'LeopardCamera1_to_LeopardCamera0_extrinsic.json'))
            json_new = {
                'image_size': intrinsic['image_size'],
                'intrinsic': intrinsic['intrinsic_matrix'],
                'radial_distortion': intrinsic['radial_distortion'],
                'tangential_distortion': intrinsic['tangential_distortion'],
                'timestamp': json_old['timestamp'],
                'LeopardCamera1_to_LeopardCamera0_extrinsic': extrinsic['extrinsic_matrix']
            }
            save_dict_as_json(json_path, json_new)

            # MEMS
            json_path = glob.glob(os.path.join(frame_folder, 'MEMS', '*.json'))[0]
            json_old = load_json(json_path)
            extrinsic1 = load_json(os.path.join(calibres_folder, 'MEMS_to_Vehicle_extrinsic.json'))
            extrinsic2 = load_json(os.path.join(calibres_folder, 'MEMS_to_VelodyneLidar_extrinsic.json'))
            json_new = {
                'MEMS_to_Vehicle_extrinsic': extrinsic1['extrinsic_matrix'],
                'MEMS_to_VelodyneLidar_extrinsic': extrinsic2['extrinsic_matrix'],
                'msg_imu': json_old['msg_imu'],
                'msg_ins': json_old['msg_ins'],
                'timestamp': json_old['timestamp']
            }
            save_dict_as_json(json_path, json_new)

            # OCULiiRadar
            json_path = glob.glob(os.path.join(frame_folder, 'OCULiiRadar', '*.json'))[0]
            json_old = load_json(json_path)
            extrinsic = load_json(os.path.join(calibres_folder, 'OCULiiRadar_to_LeopardCamera0_extrinsic.json'))
            json_new = {
                'OCULiiRadar_to_LeopardCamera0_extrinsic': extrinsic['extrinsic_matrix'],
                'timestamp': json_old['timestamp']
            }
            save_dict_as_json(json_path, json_new)

            # TIRadar
            json_path = glob.glob(os.path.join(frame_folder, 'TIRadar', '*.json'))[0]
            json_old = load_json(json_path)
            extrinsic = load_json(os.path.join(calibres_folder, 'TIRadar_to_LeopardCamera0_extrinsic.json'))
            json_new = {
                'TIRadar_to_LeopardCamera0_extrinsic': extrinsic['extrinsic_matrix'],
                'timestamp': json_old['timestamp']
            }
            save_dict_as_json(json_path, json_new)

            # VelodyneLidar
            json_path = glob.glob(os.path.join(frame_folder, 'VelodyneLidar', '*.json'))[0]
            json_old = load_json(json_path)
            extrinsic = load_json(os.path.join(calibres_folder, 'VelodyneLidar_to_LeopardCamera0_extrinsic.json'))
            json_new = {
                'VelodyneLidar_to_LeopardCamera0_extrinsic': extrinsic['extrinsic_matrix'],
                'timestamp': json_old['timestamp']
            }
            save_dict_as_json(json_path, json_new)

            log_GREEN('{}/{} {}'.format(j + 1, len(framenames), framename))



    print('done')


if __name__ == '__main__':
    main()
