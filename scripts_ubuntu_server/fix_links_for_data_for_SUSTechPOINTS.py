import os
from dataset_v2 import log_YELLOW, log_GREEN


def main():
    root_path = '/mnt/ourDataset_v2'

    data_for_SUSTechPOINTS_path = os.path.join(root_path, 'data_for_SUSTechPOINTS')

    groups = os.listdir(data_for_SUSTechPOINTS_path)
    groups.sort()
    for group in groups:
        cameras = os.listdir(os.path.join(data_for_SUSTechPOINTS_path, group, 'image'))
        for camera in cameras:
            frames = os.listdir(os.path.join(data_for_SUSTechPOINTS_path, group, 'image', camera))
            frames.sort()
            for frame in frames:
                # read link information
                src_origin = os.readlink(os.path.join(data_for_SUSTechPOINTS_path, group, 'image', camera, frame))
                src_new = src_origin.replace('/mnt/Dataset', root_path)
                if src_origin == src_new:
                    log_YELLOW('Skip {} {}'.format(group, frame))
                    continue

                dist = os.path.join(data_for_SUSTechPOINTS_path, group, 'image', camera, frame)

                # backup
                os.rename(dist, dist+'.backup')

                # modify links
                os.symlink(src_new, dist)

                # delete backup
                os.remove(dist+'.backup')

                log_GREEN('Done {} {}'.format(group, frame))




if __name__ == '__main__':
    main()
