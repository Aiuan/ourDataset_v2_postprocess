import os
import glob
import json

import numpy as np

from MEMS.asc_decoder import MEMSDecoder
from utils import *

def main():
    root_MEMS = 'F:\\20221217\\MEMS'
    output_path = 'F:\\20221217\\MEMS_json'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    asc_path = glob.glob(os.path.join(root_MEMS, '*.ASC'))[0]
    md = MEMSDecoder(asc_path)

    ts_unique = np.unique(md.data['ts_unix'].values)
    for i in range(len(ts_unique)):
        ts_str = '{:.3f}'.format(ts_unique[i])
        msg_imu, msg_ins, error, ts_str_MEMS = md.select_by_ts(ts_str)

        # save as .json
        res = {
            'msg_imu': dict([(attr, msg_imu.__getattribute__(attr)) for attr in dir(msg_imu) if not attr.startswith("__")]),
            'msg_ins': dict([(attr, msg_ins.__getattribute__(attr)) for attr in dir(msg_ins) if not attr.startswith("__")]),
        }
        path_save = os.path.join(output_path, '{}.json'.format(ts_str_MEMS))
        with open(path_save, 'w') as f:
            json.dump(res, f)

        log_GREEN('({}/{}) Save as {}'.format(i+1, len(ts_unique), path_save))


if __name__ == '__main__':
    main()
