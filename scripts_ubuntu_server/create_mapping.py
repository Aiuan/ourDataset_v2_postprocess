import os

import pandas as pd


def main():
    root_dataset = '/mnt/ourDataset_v2/ourDataset_v2'
    output_path = '/mnt/ourDataset_v2/mapping.csv'

    groups = os.listdir(root_dataset)
    groups.sort()

    # dates = []
    # group_idxs = []
    # modes = []
    # frame_idxs = []
    paths = []
    mapping_ids = []
    cnt = 0
    for i, group in enumerate(groups):
        date = group.split('_')[0]
        group_idx = int(group.split('_')[1].replace('group', ''))
        mode = group.split('_')[2]

        frames = os.listdir(os.path.join(root_dataset, group))
        frames.sort()

        for j, frame in enumerate(frames):
            frame_idx = int(frame.replace('frame', ''))
            path = os.path.join(group, frame)
            mapping_id = '{:>06d}'.format(cnt)

            # dates.append(date)
            # group_idxs.append(group_idx)
            # modes.append(mode)
            # frame_idxs.append(frame_idx)
            paths.append(path)
            mapping_ids.append(mapping_id)

            cnt += 1

    mapping = pd.DataFrame({
        'mapping_ids': pd.Series(mapping_ids, dtype='string'),
        'paths': pd.Series(paths, dtype='string'),
        # 'dates': pd.Series(dates, dtype='string'),
        # 'group_idxs': pd.Series(group_idxs, dtype='int64'),
        # 'modes': pd.Series(modes, dtype='string'),
        # 'frame_idxs': pd.Series(frame_idxs, dtype='int64')
    })

    mapping.to_csv(output_path, index=False, header=False)

    print('done')


if __name__ == '__main__':
    main()
