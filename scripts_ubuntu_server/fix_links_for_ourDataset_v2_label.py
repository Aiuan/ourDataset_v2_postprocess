import os
from dataset_v2 import log_YELLOW, log_GREEN


def main():
    root_path = '/mnt/ourDataset_v2'

    ourDataset_v2_path = os.path.join(root_path, 'ourDataset_v2')

    ourDataset_v2_label_path = os.path.join(root_path, 'ourDataset_v2_label')
    groups = os.listdir(ourDataset_v2_label_path)
    groups.sort()
    for group in groups:
        if 'mixmode' in group:
            # read link information
            src_origin = os.readlink(os.path.join(ourDataset_v2_label_path, group))
            src_new = os.path.join(ourDataset_v2_path, group)
            if src_origin == src_new:
                log_YELLOW('Skip {}'.format(group))
                continue

            dist = os.path.join(ourDataset_v2_label_path, group)

            # backup
            os.rename(dist, dist + '.backup')

            # modify links
            os.symlink(src_new, dist)

            # delete backup
            os.remove(dist + '.backup')

            log_GREEN('Done {}'.format(group))

        else:
            frames = os.listdir(os.path.join(ourDataset_v2_label_path, group))
            frames.sort()
            for frame in frames:
                # read link information
                src_origin = os.readlink(os.path.join(ourDataset_v2_label_path, group, frame))
                src_new = os.path.join(ourDataset_v2_path, group, frame)
                if src_origin == src_new:
                    log_YELLOW('Skip {} {}'.format(group, frame))
                    continue

                dist = os.path.join(ourDataset_v2_label_path, group, frame)

                # backup
                os.rename(dist, dist+'.backup')

                # modify links
                os.symlink(src_new, dist)

                # delete backup
                os.remove(dist+'.backup')

                log_GREEN('Done {} {}'.format(group, frame))




if __name__ == '__main__':
    main()
