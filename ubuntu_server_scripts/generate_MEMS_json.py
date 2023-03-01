import os
import json
import argparse
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np

from MEMS.asc_decoder import MEMSDecoder
from utils import *

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, help='MEMS asc path')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    asc_path = args.data_path  # 'F:\\20221217\\MEMS\\NMUT21160006Z_2022-12-17_09-17-59.ASC'
    output_path = '{}_json'.format(os.path.dirname(asc_path))  # 'F:\\20221217\\MEMS_json'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    md = MEMSDecoder(asc_path)

    ts_unique = np.unique(md.data['ts_unix'].values)
    for i in range(len(ts_unique)):
        ts_str = '{:.3f}'.format(ts_unique[i])
        msg_imu, msg_ins, error, ts_str_MEMS = md.select_by_ts(ts_str)

        path_save = os.path.join(output_path, '{}.json'.format(ts_str_MEMS))

        if msg_ins.ins_status != 'INS_SOLUTION_GOOD':
            log_YELLOW('msg_ins.ins_status = {}'.format(msg_ins.ins_status))

        if msg_ins.pos_type != 'INS_RTKFIXED':
            log_YELLOW('msg_ins.pos_type = {}'.format(msg_ins.pos_type))

        # save as .json
        res = {
            'msg_imu': dict([(attr, msg_imu.__getattribute__(attr)) for attr in dir(msg_imu) if not attr.startswith("__")]),
            'msg_ins': dict([(attr, msg_ins.__getattribute__(attr)) for attr in dir(msg_ins) if not attr.startswith("__")]),
        }

        with open(path_save, 'w') as f:
            json.dump(res, f)
        log_GREEN('({}/{}) Save as {}'.format(i+1, len(ts_unique), path_save))


if __name__ == '__main__':
    main()
