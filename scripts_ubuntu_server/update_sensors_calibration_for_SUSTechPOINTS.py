import os
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np

from dataset_v2 import log, log_GREEN, load_json, save_dict_as_json

def main():
    root_data = '/mnt/Dataset/data_for_SUSTechPOINTS'
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
    for i, groupname in enumerate(groupnames):
        log('='*100)
        log('{}/{} {}'.format(i+1, len(groupnames), groupname))

        group_folder = os.path.join(root_data, groupname)

        exp_name = groupname.split('_')[0]
        calibres_name = map[exp_name]
        calibres_folder = os.path.join(root_calib, calibres_name)

        calib_folder = os.path.join(group_folder, 'calib')
        items = os.listdir(calib_folder)
        for item in items:

            camera_name = item.replace('.json', '')

            intrinsic = load_json(os.path.join(calibres_folder, '{}_intrinsic.json'.format(camera_name)))
            intrinsic = np.array(intrinsic['intrinsic_matrix'])
            intrinsic = intrinsic.reshape((-1)).tolist()

            extrinsic0 = load_json(os.path.join(calibres_folder, 'VelodyneLidar_to_LeopardCamera0_extrinsic.json'))
            extrinsic0 = np.array(extrinsic0['extrinsic_matrix'])
            if camera_name == 'LeopardCamera0':
                extrinsic = extrinsic0
                extrinsic = extrinsic.reshape((-1)).tolist()
            else:
                extrinsic_to0 = load_json(os.path.join(calibres_folder, '{}_to_LeopardCamera0_extrinsic.json'.format(camera_name)))
                extrinsic_to0 = np.array(extrinsic_to0['extrinsic_matrix'])
                extrinsic = np.matmul(np.linalg.inv(extrinsic_to0),  extrinsic0)
                extrinsic = extrinsic.reshape((-1)).tolist()

            # update
            item_path = os.path.join(calib_folder, item)
            save_dict_as_json(
                item_path,
                {
                    'intrinsic': intrinsic,
                    'extrinsic': extrinsic
                }
            )

            log_GREEN('update {}.json'.format(camera_name))

    print('done')


if __name__ == '__main__':
    main()
