import os
import sys

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import matplotlib.pyplot as plt

from dataset_v2 import *

def main():
    root_group = '/mnt/Dataset/ourDataset_v2/20221217_group0000_mode1_279frames'
    frame_idx = 0

    group = Group(root_group)
    frame = group[frame_idx]
    route = group.get_route()

    MEMS_json = frame.get_sensor_data('MEMS_json')
    MEMS_ts_str = MEMS_json['timestamp']
    MEMS_localtime = unix2local(MEMS_ts_str)
    MEMS_height, MEMS_width = 512, 640

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(route['longitude_E'].values, route['latitude_N'].values, '-', c='k', linewidth=1)
    ax.scatter(route['longitude_E'].values, route['latitude_N'].values, s=1, c=np.round(route.index.values), cmap='jet')
    ax.scatter(MEMS_json['msg_ins']['longitude_E'], MEMS_json['msg_ins']['latitude_N'], s=100, c='k', marker='P')
    ax.text(
        MEMS_json['msg_ins']['longitude_E'] + 1e-5, MEMS_json['msg_ins']['latitude_N'],
        '{:.6f}N, {:.6f}E'.format(MEMS_json['msg_ins']['longitude_E'], MEMS_json['msg_ins']['latitude_N'])
    )

    delta_x = route['longitude_E'].values.max() - route['longitude_E'].values.min()
    delta_y = route['latitude_N'].values.max() - route['latitude_N'].values.min()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.grid(color='gray', linestyle='-', linewidth=1)
    ax.set_aspect(MEMS_height/MEMS_width*delta_x/delta_y)

    ax.set_title('MEMS\n{}'.format(MEMS_localtime))

    plt.show()


    print('done')


if __name__ == '__main__':
    main()

