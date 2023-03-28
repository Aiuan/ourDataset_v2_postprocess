import os
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

from openpyxl import load_workbook
import numpy as np
import pandas as pd

from utils import *

def read_workbook(wb_path, sheetIndex=0):
    # open file
    wb = load_workbook(wb_path, data_only=True)
    worksheets_name_list = wb.sheetnames
    ws = wb[worksheets_name_list[sheetIndex]]
    data = ws.values
    cols = next(data)
    data = list(data)
    df = pd.DataFrame(data, columns=cols)
    return df

def main():
    info_path = '/mnt/Dataset/ourDataset_v2_group_infomation.xlsx'
    label_per_frames = 5
    dataset_v2_path = '/mnt/Dataset/ourDataset_v2'
    output_path = '/mnt/Dataset/ourDataset_v2_label'

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        log_GREEN('Create {}'.format(output_path))

    df_info = read_workbook(info_path)

    keys_dict = dict()
    for key in df_info.columns.values.tolist():
        keys_dict[key.split('\n')[0]] = key

    num_groups_label = df_info[keys_dict['need_to_label']].values.sum()
    log_BLUE('Need to label {} groups'.format(num_groups_label))

    # calculate number of frame to label
    df_info['num_frames'] = df_info[keys_dict['group_name']].str.split('_', expand=True)[3].str.replace('frames', '').astype('int')
    df_info['num_frames_to_label'] = 1 + np.floor(df_info['num_frames'].values * 1.0 / label_per_frames).astype('int')
    df_info['idx_frame_label'] = [np.arange(0, num_frames, label_per_frames) for num_frames in df_info['num_frames'].values]

    log_BLUE('Need to label {} frames'.format(df_info['num_frames_to_label'].values.sum()))

    cnt_groups_label = 0
    for i in range(df_info.shape[0]):
        if not df_info[keys_dict['need_to_label']].iloc[i]:
            log_YELLOW('Skip {} by manual selection'.format(df_info[keys_dict['group_name']].iloc[i]))
            continue

        cnt_groups_label += 1
        cnt_frames_label = 0
        group_folder = os.path.join(dataset_v2_path, df_info[keys_dict['group_name']].iloc[i])
        for idx_frame in df_info['idx_frame_label'].iloc[i]:
            log('='*100)

            cnt_frames_label += 1
            num_frames_in_group = df_info['num_frames'].iloc[i]
            num_frames_label = df_info['idx_frame_label'].iloc[i].shape[0]
            frame_foldername = 'frame{:>04d}'.format(idx_frame)
            frame_folder = os.path.join(group_folder, frame_foldername)

            # build link
            src_path = frame_folder
            dst_path = os.path.join(output_path, df_info[keys_dict['group_name']].iloc[i], frame_foldername)
            if not os.path.exists(os.path.dirname(dst_path)):
                os.mkdir(os.path.dirname(dst_path))
                log_GREEN('Create {}'.format(os.path.dirname(dst_path)))
            try:
                os.remove(dst_path)
                log_YELLOW('Remove {}'.format(dst_path))
            except:
                pass
            os.symlink(src_path, dst_path)
            log_GREEN('Link {}'.format(dst_path))
            log_BLUE(
                'Group in ourDataset_v2: {}/{}, Group to label: {}/{}, Idx_frame in Group: {}/{}, Frame to label: {}/{}'.format(
                    i+1, df_info.shape[0], cnt_groups_label, num_groups_label,
                    idx_frame, num_frames_in_group-1, cnt_frames_label, num_frames_label
                )
            )


    log('done')

if __name__ == '__main__':
    main()

