import os
import sys
import argparse

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import pandas as pd
import matplotlib.pyplot as plt

from dataset_v2 import log, load_json

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_label', type=str, default='/mnt/Dataset/labels_for_checking2', help='labelroot path')
    parser.add_argument('--label_foldername', type=str, default='', help='label folder name')
    parser.add_argument('--output_foldername', type=str, default='runs2/distribution', help='output folder name')

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

    for label_foldername in label_foldernames:

        # check and create output folder
        root_output = os.path.join(CURRENT_ROOT, output_foldername, label_foldername)
        if not os.path.exists(root_output):
            os.makedirs(root_output)
            log('create {}'.format(root_output))

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

        # check lwh distribution in each class
        df_lwh = df_package.groupby(['class'])
        for name, group in df_lwh:
            print(name)

            plt.hist(group['l'].values, bins=10)
            plt.axvline(
                x=group['l'].mean(), color='red', linestyle='--',
                label='mean={:.3f}\nstd={:.3f}'.format(
                    group['l'].mean(), group['l'].std()
                )
            )
            plt.title('{}-length'.format(name))
            plt.legend()
            plt.savefig(os.path.join(root_output, '{}-{}-length.png'.format(label_foldername, name)))
            # plt.show()
            plt.close()

            plt.hist(group['w'].values, bins=10)
            plt.axvline(
                x=group['w'].mean(), color='red', linestyle='--',
                label='mean={:.3f}\nstd={:.3f}'.format(
                    group['w'].mean(), group['w'].std()
                )
            )
            plt.title('{}-width'.format(name))
            plt.legend()
            plt.savefig(os.path.join(root_output, '{}-{}-width.png'.format(label_foldername, name)))
            # plt.show()
            plt.close()

            plt.hist(group['h'].values, bins=10)
            plt.axvline(
                x=group['h'].mean(), color='red', linestyle='--',
                label='mean={:.3f}\nstd={:.3f}'.format(
                    group['h'].mean(), group['h'].std()
                )
            )
            plt.title('{}-height'.format(name))
            plt.legend()
            plt.savefig(os.path.join(root_output, '{}-{}-height.png'.format(label_foldername, name)))
            # plt.show()
            plt.close()

            group.to_excel(os.path.join(root_output, '{}-{}.xlsx'.format(label_foldername, name)))


    print('done')


if __name__ == '__main__':
    main()

