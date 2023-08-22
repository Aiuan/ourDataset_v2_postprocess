import os
import sys
import glob

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

from dataset_v2 import log, log_GREEN, load_json, save_dict_as_json, log_YELLOW

def main():
    root_dataset = '/mnt/ourDataset_v2/ourDataset_v2_label'
    root_label = '/mnt/ourDataset_v2/raw_data/labels'

    groupnames = os.listdir(root_dataset)
    groupnames.sort()

    packagenames = os.listdir(root_label)
    packagenames.sort(key=lambda x: int(x.replace('package', '')))
    package_group_map = dict()
    log('=' * 100)
    for i, packagename in enumerate(packagenames):
        items = os.listdir(os.path.join(root_label, packagename))
        items.sort()
        log("{}/{} {} include: {} groups' label".format(i + 1, len(packagenames), packagename, len(items)))
        for item in items:
            package_group_map[item] = packagename


    for i, groupname in enumerate(groupnames):
        log('='*100)
        log('{}/{} {}'.format(i+1, len(groupnames), groupname))

        root_group_data = os.path.join(root_dataset, groupname)
        if not os.path.exists(root_group_data):
            log_YELLOW('Not exist {}'.format(root_group_data))
            exit()

        root_group_label = os.path.join(root_label, package_group_map[groupname], groupname)
        framenames = [item.split('.')[0] for item in os.listdir(root_group_label)]
        framenames.sort()
        for j, framename in enumerate(framenames):
            root_frame_data = os.path.join(root_group_data, framename)
            if not os.path.exists(root_frame_data):
                log_YELLOW('Not exist {}'.format(root_frame_data))
                exit()

            path_json = glob.glob(os.path.join(root_frame_data, 'VelodyneLidar', '*.json'))[0]
            data = load_json(path_json)

            path_annotation = os.path.join(root_group_label, '{}.json'.format(framename))
            annotation = load_json(path_annotation)

            data['annotation'] = annotation

            save_dict_as_json(path_json, data)
            log_GREEN('{}/{} {} merge label successfully'.format(j+1, len(framenames), framename))


if __name__ == '__main__':
    main()

