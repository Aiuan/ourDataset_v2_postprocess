import os
import sys
import argparse
import re
from functools import cmp_to_key

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import pandas as pd

from dataset_v2 import log, load_json

def numerical_sort(value1, value2):
    pattern = re.compile('(\d+)')
    match1 = pattern.search(value1)
    match2 = pattern.search(value2)
    if match1 and match2:
        return int(match1.group()) - int(match2.group())
    elif match1:
        return -1
    elif match2:
        return 1
    else:
        return 0

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_label', type=str, default='/mnt/Dataset/labels_for_checking2', help='labelroot path')
    parser.add_argument('--label_foldername', type=str, default='', help='label folder name')
    parser.add_argument('--output_foldername', type=str, default='runs2/summary', help='output folder name')

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
        label_foldernames = sorted(os.listdir(root_label), key=cmp_to_key(numerical_sort))

    root_output = os.path.join(CURRENT_ROOT, output_foldername)
    if not os.path.exists(root_output):
        os.makedirs(root_output)
        log('create {}'.format(root_output))

    df_summary = pd.DataFrame()
    summary_path = os.path.join(root_output, 'summary.xlsx')

    for label_foldername in label_foldernames:
        # check whether has been processed, if processed, just read
        output_path = os.path.join(root_output, '{}.xlsx'.format(label_foldername))
        if os.path.exists(output_path):
            df_package = pd.read_excel(output_path, index_col=0)
        else:
            df_package = pd.DataFrame()
            label_folder = os.path.join(root_label, label_foldername)
            groupnames = os.listdir(label_folder)
            groupnames.sort()
            for i, groupname in enumerate(groupnames):
                log('='*100)
                log('{}/{} {}'.format(i+1, len(groupnames), groupname))

                df_group= pd.DataFrame()
                group_folder = os.path.join(label_folder, groupname)

                framenames = os.listdir(group_folder)
                framenames = [framename.split('.')[0] for framename in framenames]
                framenames.sort()

                for j, framename in enumerate(framenames):
                    log('{}/{} {}'.format(j+1, len(framenames), framename))

                    labels = load_json(os.path.join(group_folder, '{}.json'.format(framename)))

                    df_frame = pd.DataFrame(labels)
                    df_frame['framename'] = framename
                    df_group = pd.concat((df_group, df_frame))

                df_group['groupname'] = groupname
                df_package = pd.concat((df_package, df_group))
            df_package.to_excel(output_path)

        # check number of bbox
        group_counts = df_package["class"].value_counts()
        group_df = pd.DataFrame(group_counts).T
        group_df.rename(index={'class': label_foldername}, inplace=True)
        df_summary = pd.concat((df_summary, group_df))

    df_summary.fillna(0, inplace=True)

    col_sum = df_summary.sum(axis=1)
    df_summary['sum'] = col_sum

    row_sum = pd.DataFrame(df_summary.sum(axis=0)).T
    row_sum.rename(index={0: 'sum'}, inplace=True)
    df_summary = pd.concat((df_summary, row_sum))

    df_summary.to_excel(summary_path)


    print('done')


if __name__ == '__main__':
    main()

