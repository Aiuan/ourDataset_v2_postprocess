import json
import os
import numpy as np
import pandas as pd

class MEMSDecoder(object):
    def __init__(self, asc_path):
        self.asc_path = asc_path
        self.data = pd.read_csv(self.asc_path, sep=' ', header=None, names=['msg'])

        header = self.data['msg'].str.split(';', expand=True)[0]
        header = header.str.split(',', expand=True)
        week = header[5].values.astype('int32')
        second = header[6].values.astype('float64')
        ts_gps = week * 7 * 24 * 3600 + second
        ts_unix = ts_gps + 315964800 - 18
        self.data.insert(loc=0, column='ts_unix', value=ts_unix)

        msg_type = header[0].values
        self.data.insert(loc=1, column='msg_type', value=msg_type)

    def select_by_ts(self, ts_str):
        ts_unix_unique = np.unique(self.data['ts_unix'].values)
        tmp = float(ts_str) - ts_unix_unique
        idx = np.argmin(np.abs(tmp))
        error = tmp[idx]
        ts_unix_nearest = ts_unix_unique[idx]
        ts_str_matched = '{:.3f}'.format(ts_unix_nearest)

        res = self.data[self.data['ts_unix'] == ts_unix_nearest]
        assert len(res) == 2
        for i in range(len(res)):
            assert res['msg_type'].iloc[i] == '#RAWIMUA' or res['msg_type'].iloc[i] == '#INSPVAXA'
            if res['msg_type'].iloc[i] == '#RAWIMUA':
                msg_imu = MsgImu(res['msg'].iloc[i])
            else:
                msg_ins = MsgIns(res['msg'].iloc[i])

        return msg_imu, msg_ins, error, ts_str_matched

class MsgIns(object):
    def __init__(self, msg):
        tmp = msg.split(';')
        header = tmp[0]
        tmp = tmp[1].split('*')
        tail = tmp[1]
        fields = tmp[0].split(',')

        # decode fields
        self.ins_status = fields[0]
        assert self.ins_status == 'INS_SOLUTION_GOOD'
        self.pos_type = fields[1]
        assert self.pos_type == 'INS_RTKFIXED'
        self.latitude_N = float(fields[2])  # degrees
        self.longitude_E = float(fields[3])  # degrees
        self.height = float(fields[4])  # m
        self.undulation = float(fields[5])  # m
        self.north_vel = float(fields[6])  # m/s
        self.east_vel = float(fields[7])  # m/s
        self.up_vel = float(fields[8])  # m/s
        self.roll = float(fields[9])  # degrees
        self.pitch = float(fields[10])  # degrees
        self.azimuth = float(fields[11])  # degrees

        # standard deviation
        self.latitude_N_dev = float(fields[12])  # degrees
        self.longitude_E_dev = float(fields[13])  # degrees
        self.height_dev = float(fields[14])  # m
        self.north_vel_dev = float(fields[15])  # m/s
        self.east_vel_dev = float(fields[16])  # m/s
        self.up_vel_dev = float(fields[17])  # m/s
        self.roll_dev = float(fields[18])  # degrees
        self.pitch_dev = float(fields[19])  # degrees
        self.azimuth_dev = float(fields[20])  # degrees


class MsgImu(object):
    def __init__(self, msg):
        tmp = msg.split(';')
        header = tmp[0]
        tmp = tmp[1].split('*')
        tail = tmp[1]
        fields = tmp[0].split(',')

        # decode fields
        self.week = int(fields[0])
        self.seconds_into_week = float(fields[1])
        self.imu_status = fields[2]

        update_rate = 20  # Hz
        accel_scale_factor = (0.400 / 65536) / 200  # mG/s/LSB
        # Change in velocity count along z axis
        self.z_accelerated_velocity = float(fields[3]) * accel_scale_factor * update_rate * 9.8e-3  # m/s^2
        self.y_accelerated_velocity = -float(fields[4]) * accel_scale_factor * update_rate * 9.8e-3  # m/s^2
        self.x_accelerated_velocity = float(fields[5]) * accel_scale_factor * update_rate * 9.8e-3  # m/s^2

        gyro_scale_factor = (0.0151515 / 65536) / 200  # deg/LSB
        # Change in angle count around z axis Right-handed
        self.z_angular_velocity = float(fields[6]) * gyro_scale_factor * update_rate  # deg/s
        self.y_angular_velocity = -float(fields[7]) * gyro_scale_factor * update_rate  # deg/s
        self.x_angular_velocity = float(fields[8]) * gyro_scale_factor * update_rate  # deg/s


def test():
    asc_path = 'F:\\20221217\\MEMS\\NMUT21160006Z_2022-12-17_09-17-59.ASC'
    md = MEMSDecoder(asc_path)

    ts_str = '1671267290.996'
    msg_imu, msg_ins, error, ts_str_matched = md.select_by_ts(ts_str)

    print('done')

def test_read():
    json_path = 'F:\\20221217_process\\MEMS\\20221217_165447_mode1_682\\1671267291.000.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    print('done')

if __name__ == '__main__':
    # test()
    test_read()
