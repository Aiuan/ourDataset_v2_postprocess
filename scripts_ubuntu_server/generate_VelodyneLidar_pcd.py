import os
import argparse
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

from VelodyneLidar.decoder import *

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='./VelodyneLidar/Alpha Prime.xml', help='lidar config file path')
    parser.add_argument('--data_path', type=str, help='lidar .pcap file path')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    config_path = args.config_path
    data_path = args.data_path  # F:\\20221217\\VelodyneLidar\\lidardata_20221217_1718.pcap
    output_path = '{}_pcd'.format(os.path.dirname(data_path))  # F:\\20221217\\VelodyneLidar_pcd

    vd = VelodyneDecoder(config_path=config_path, pcap_path=data_path, output_path=output_path)

    t_last = time.time()
    while 1:
        vd.decode_next_packet()
        if vd.judge_jump_cut_degree():
            vd.generate_frame(pcd_file_type='pcd')
            t = time.time()
            print('    {:.2f} s'.format(t - t_last))
            t_last = time.time()


if __name__ == '__main__':
    main()
