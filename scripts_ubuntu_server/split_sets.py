import os

import numpy as np
import pandas as pd


def get_valid_groups(root_dataset):
    groups = os.listdir(root_dataset)
    groups.sort()

    valid_groups = [group for group in groups if 'mixmode' not in group]
    valid_groups.sort()

    return valid_groups


def split_sets(root_dataset, groups, train_ratio, valid_ratio, test_ratio):
    groups_train = []
    cnt_frames_train = 0
    groups_valid = []
    cnt_frames_valid = 0
    groups_test = []
    cnt_frames_test = 0

    # split groups by date
    splited_groups = {}
    for group in groups:
        date = group.split('_')[0]
        if date not in splited_groups.keys():
            splited_groups[date] = []
        splited_groups[date].append(group)

    # split groups in single date
    for date, groups_in_single_date in splited_groups.items():
        (this_groups_train, this_groups_valid, this_groups_test,
         this_cnt_frames_train, this_cnt_frames_valid, this_cnt_frames_test) = split_groups_in_single_date(
            root_dataset=root_dataset,
            date=date,
            groups=groups_in_single_date,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )
        groups_train.extend(this_groups_train)
        groups_valid.extend(this_groups_valid)
        groups_test.extend(this_groups_test)
        cnt_frames_train += this_cnt_frames_train
        cnt_frames_valid += this_cnt_frames_valid
        cnt_frames_test += this_cnt_frames_test

    print(f'Split results:')
    print(f'train: {len(groups_train)} groups, {cnt_frames_train} frames')
    print(f'valid: {len(groups_valid)} groups, {cnt_frames_valid} frames')
    print(f'test: {len(groups_test)} groups, {cnt_frames_test} frames')

    return groups_train, groups_valid, groups_test, cnt_frames_train, cnt_frames_valid, cnt_frames_test


def split_groups_in_single_date(root_dataset, date, groups, train_ratio, valid_ratio, test_ratio):
    # calculate total number of labeled frames
    num_frames = np.array(
        [len(os.listdir(os.path.join(root_dataset, group))) for group in groups],
        dtype=np.int64
    )
    num_frames_sum = num_frames.sum()

    # desired number of frames in each set
    num_frames_train = round(num_frames_sum * train_ratio / (train_ratio + valid_ratio + test_ratio))
    num_frames_valid = round(num_frames_sum * valid_ratio / (train_ratio + valid_ratio + test_ratio))
    num_frames_test = round(num_frames_sum * test_ratio / (train_ratio + valid_ratio + test_ratio))

    num_frames_cumulative_sum = np.cumsum(num_frames)

    start_train = 0
    end_train = np.where(num_frames_cumulative_sum >= num_frames_train)[0][0] + 1

    start_valid = end_train
    end_valid = np.where(num_frames_cumulative_sum >= (num_frames_train + num_frames_valid))[0][0] + 1

    start_test = end_valid
    end_test = len(groups)

    # splited results
    groups_train = groups[start_train:end_train]
    groups_valid = groups[start_valid:end_valid]
    groups_test = groups[start_test:end_test]

    # actual number of frames in each set
    cnt_frames_train = num_frames_cumulative_sum[end_train - 1]
    cnt_frames_valid = num_frames_cumulative_sum[end_valid - 1] - num_frames_cumulative_sum[end_train - 1]
    cnt_frames_test = num_frames_cumulative_sum[end_test - 1] - num_frames_cumulative_sum[end_valid - 1]

    print(
        f'{num_frames_sum} frames in {date}, split them to train({cnt_frames_train}), valid({cnt_frames_valid}), test({cnt_frames_test})')

    return groups_train, groups_valid, groups_test, cnt_frames_train, cnt_frames_valid, cnt_frames_test


def main():
    root_dataset = '/mnt/ourDataset_v2/ourDataset_v2_label'
    root_output = '/mnt/ourDataset_v2'
    path_train = os.path.join(root_output, 'train.txt')
    path_valid = os.path.join(root_output, 'valid.txt')
    path_trainval = os.path.join(root_output, 'trainval.txt')
    path_test = os.path.join(root_output, 'test.txt')
    path_mapping = '/mnt/ourDataset_v2/mapping.csv'

    train_ratio = 3
    valid_ratio = 1
    test_ratio = 0

    valid_groups = get_valid_groups(root_dataset=root_dataset)

    '''
        每天总帧数的
            约 train_ratio / (train_ratio + valid_ratio + test_ratio) 用于训练集，
            约 valid_ratio / (train_ratio + valid_ratio + test_ratio) 用于验证集，
            约  test_ratio / (train_ratio + valid_ratio + test_ratio) 用于测试集
            保证不分割一个group中的frames，分别存在于训练集/验证集/测试集中
        日期      地点    时间
        201217    舟山    傍晚
        201219    舟山    夜晚
        201220    舟山    白天
        201221    舟山    白天
        201223    烟台    夜晚
        201224    烟台    白天
    '''
    groups_train, groups_valid, groups_test, cnt_frames_train, cnt_frames_valid, cnt_frames_test = split_sets(
        root_dataset=root_dataset,
        groups=valid_groups,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio
    )

    '''use mapping and dump splited results to txt'''
    # load mapping
    mapping = pd.read_csv(path_mapping, names=['mapping_ids', 'paths'], dtype='string')

    # get frames' mapping_id in each set
    mapping_ids_train = []
    mapping_ids_valid = []
    mapping_ids_trainval = []
    mapping_ids_test = []
    for group in groups_train:
        frames = os.listdir(os.path.join(root_dataset, group))
        frames.sort()
        for frame in frames:
            path = os.path.join(group, frame)
            mapping_id = mapping[mapping['paths'] == path]['mapping_ids'].values[0]
            mapping_ids_train.append(mapping_id)
            mapping_ids_trainval.append(mapping_id)

    for group in groups_valid:
        frames = os.listdir(os.path.join(root_dataset, group))
        frames.sort()
        for frame in frames:
            path = os.path.join(group, frame)
            mapping_id = mapping[mapping['paths'] == path]['mapping_ids'].values[0]
            mapping_ids_valid.append(mapping_id)
            mapping_ids_trainval.append(mapping_id)

    for group in groups_test:
        frames = os.listdir(os.path.join(root_dataset, group))
        frames.sort()
        for frame in frames:
            path = os.path.join(group, frame)
            mapping_id = mapping[mapping['paths'] == path]['mapping_ids'].values[0]
            mapping_ids_test.append(mapping_id)

    # dump to txt
    with open(path_train, 'w') as f:
        f.writelines([line + '\n' for line in mapping_ids_train])

    with open(path_valid, 'w') as f:
        f.writelines([line + '\n' for line in mapping_ids_valid])

    with open(path_trainval, 'w') as f:
        f.writelines([line + '\n' for line in mapping_ids_trainval])

    with open(path_test, 'w') as f:
        f.writelines([line + '\n' for line in mapping_ids_test])

    print('done')


if __name__ == '__main__':
    main()
