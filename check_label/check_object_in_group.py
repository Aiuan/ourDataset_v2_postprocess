import os
import sys
import argparse

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import pandas as pd

from dataset_v2 import log, load_json

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_label', type=str, default='/mnt/Dataset/labels_for_checking2', help='labelroot path')
    parser.add_argument('--label_foldername', type=str, default='', help='label folder name')
    parser.add_argument('--output_foldername', type=str, default='runs2/object', help='output folder name')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    root_label = args.root_label
    label_foldername = args.label_foldername
    output_foldername = args.output_foldername

    if label_foldername != '':
        label_foldernames = [label_foldername]
    else:
        label_foldernames = os.listdir(root_label)
        label_foldernames.sort()

    root_output = os.path.join(CURRENT_ROOT, output_foldername)
    if not os.path.exists(root_output):
        os.makedirs(root_output)
        log('create {}'.format(root_output))

    for label_foldername in label_foldernames:
        label_folder = os.path.join(root_label, label_foldername)
        groupnames = os.listdir(label_folder)
        groupnames.sort()
        for i, groupname in enumerate(groupnames):
            log('=' * 100)
            log('{}/{} {}'.format(i + 1, len(groupnames), groupname))

            output_folder = os.path.join(root_output, '{}-{}'.format(label_foldername, groupname))
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
                log('create {}'.format(output_folder))

            df_group_path = os.path.join(output_folder, 'df_group.xlsx')
            if os.path.exists(df_group_path):
                df_group = pd.read_excel(df_group_path, index_col=0)
            else:
                df_group = pd.DataFrame()
                group_folder = os.path.join(label_folder, groupname)
                framenames = os.listdir(group_folder)
                framenames = [framename.split('.')[0] for framename in framenames]
                framenames.sort()
                for j, framename in enumerate(framenames):
                    log('{}/{} {}'.format(j + 1, len(framenames), framename))

                    labels = load_json(os.path.join(group_folder, '{}.json'.format(framename)))

                    df_frame = pd.DataFrame(labels)
                    df_frame['framename'] = framename
                    df_group = pd.concat((df_group, df_frame))
                df_group['groupname'] = groupname
                df_group.to_excel(df_group_path)

            # check each object
            for object_id, df_object in df_group.groupby(['object_id']):
                print(object_id)

                msg = ''

                # judge object's lwh
                if df_object['l'].std() > 0.1:
                    msg = msg + 'l'
                if df_object['w'].std() > 0.1:
                    msg = msg + 'w'
                if df_object['h'].std() > 0.1:
                    msg = msg + 'h'

                df_object.to_excel(os.path.join(output_folder, 'object{}_{}.xlsx'.format(object_id, msg)))


    print('done')


if __name__ == '__main__':
    main()

