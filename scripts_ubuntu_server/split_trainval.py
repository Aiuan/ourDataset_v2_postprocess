import os

import numpy as np
import pandas as pd

def main():
    root_dataset = '/mnt/ourDataset_v2/ourDataset_v2_label'
    output_train = '/mnt/ourDataset_v2/train.txt'
    output_val = '/mnt/ourDataset_v2/val.txt'
    output_trainval = '/mnt/ourDataset_v2/trainval.txt'
    train_val_rate = 0.75  # train:val = 3:1

    groups_cared = [group for group in os.listdir(root_dataset) if 'mixmode' not in group]
    groups_cared.sort()

    days = []
    groups_id = []
    frames_id = []
    frames_cared = []
    for i, group in enumerate(groups_cared):
        day = group.split('_')[0]

        frames = os.listdir(os.path.join(root_dataset, group))
        frames.sort()

        days.extend([day] * len(frames))
        groups_id.extend([i] * len(frames))
        frames_id.extend(list(range(len(frames))))
        frames_cared.extend([os.path.join(group, frame) for frame in frames])

    df = pd.DataFrame({
        'days': days,
        'groups_id': groups_id,
        'frames_id': frames_id,
        'path': frames_cared
    })

    '''
        每天总帧数的约（train_val_rate）%用于训练集，剩下的用于验证集
        约（train_val_rate）%： 保证不分割一个group中的frames，分别存在于训练集和验证集
        日期      地点    时间
        201217    舟山    傍晚
        201219    舟山    夜晚
        201220    舟山    白天
        201221    舟山    白天
        201223    舟山    夜晚
        201224    舟山    白天
    '''
    frames_in_train = []
    frames_in_val = []
    frames_in_trainval = df['path'].tolist()

    for day, df_day in df.groupby('days'):
        thred = round(df_day.shape[0] * train_val_rate)

        cnt = 0  # number of frames in trainset
        for group, df_group in df_day.groupby('groups_id'):
            if cnt <= thred:
                if cnt + df_group.shape[0] <= thred:
                    # add to trainset
                    frames_in_train.extend(df_group['path'].tolist())
                    cnt += df_group.shape[0]
                else:
                    if (thred - cnt) > (cnt + df_group.shape[0] - thred):
                        # add to trainset
                        frames_in_train.extend(df_group['path'].tolist())
                        cnt += df_group.shape[0]
                    else:
                        # add to valset
                        frames_in_val.extend(df_group['path'].tolist())
            else:
                # add to valset
                frames_in_val.extend(df_group['path'].tolist())

        print('In day {}'.format(day), end=', ')
        print('{} frames'.format(df_day.shape[0]), end=', ')
        print('{} frames in trainset'.format(cnt), end=', ')
        print('{} frames in valset'.format(df_day.shape[0] - cnt))

    print('Total information:')
    print('The number of frames in train: {}'.format(len(frames_in_train)))
    print('The number of frames in val: {}'.format(len(frames_in_val)))
    print('The number of frames in trainval: {}'.format(len(frames_in_trainval)))

    # write to txt
    with open(output_train, 'w') as f:
        f.writelines([line + '\n' for line in frames_in_train])

    with open(output_val, 'w') as f:
        f.writelines([line + '\n' for line in frames_in_val])

    with open(output_trainval, 'w') as f:
        f.writelines([line + '\n' for line in frames_in_trainval])


    print('done')

if __name__ == '__main__':
    main()
