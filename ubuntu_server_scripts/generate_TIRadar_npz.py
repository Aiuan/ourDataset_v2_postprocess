import os
import argparse
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

from TIRadar.adcdata_decoder import RecordData_NormalMode, RecordData_MixMode

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, help='TIRadar folder path')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    root = args.data_path  # 'F:\\20221217\\TIRadar'
    output_root = '{}_npz'.format(root)  # 'F:\\20221217\\TIRadar_npz'
    for item in os.listdir(root):
        if 'mixmode' in item:
            record_data = RecordData_MixMode(os.path.join(root, item))
        else:
            record_data = RecordData_NormalMode(os.path.join(root, item))
        output_folder = os.path.join(output_root, record_data.folder)
        record_data.divide_frame(output_folder)

    print('done')

if __name__ == '__main__':
    main()
