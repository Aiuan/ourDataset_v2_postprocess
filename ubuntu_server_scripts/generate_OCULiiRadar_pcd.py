import os
import time
import argparse
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

from OCUliiRadar.decoder_np import OCULiiDecoderNetworkPackets

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, help='OCULiiRadar .pcap path')

    args = parser.parse_args()
    return args

def decode_network_packets():
    args = get_args()

    data_path = args.data_path  # 'F:\\20221217\\OCULiiRadar\\20221217_1\\20221217_1_udp.pcap'
    output_path = '{}_pcd'.format(os.path.dirname(os.path.dirname(data_path)))  # 'F:\\20221217\\OCULiiRadar_pcd'

    odnp = OCULiiDecoderNetworkPackets(pcap_path=data_path, output_path=output_path, pcd_file_type='pcd')

    t_last = time.time()
    while 1:
        odnp.decode()
        t = time.time()
        print('    {:.2f} s'.format(t - t_last))
        print('='*100)
        t_last = time.time()

if __name__ == '__main__':
    decode_network_packets()
