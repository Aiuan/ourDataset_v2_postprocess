import os
import time
import pandas as pd
import numpy as np
import dpkt
import socket
import xml.etree.ElementTree as ET

from utils import *

class LidarCalib(object):
    def __init__(self, config_path):
        self.config_path = config_path
        tree = ET.parse(config_path)
        root = tree.getroot()
        n = int(root.find('DB').find('enabled_').find('count').text)
        self.n_lasers = n

        val = []
        for i in range(n):
            val.append({})

        for i, d in enumerate(root.find('DB').find('enabled_').findall('item')):
            val[i]['min_intensity'] = int(d.text)

        for i, d in enumerate(root.find('DB').find('minIntensity_').findall('item')):
            val[i]['min_intensity'] = int(d.text)

        for i, d in enumerate(root.find('DB').find('maxIntensity_').findall('item')):
            val[i]['max_intensity'] = int(d.text)

        for i, d in enumerate(root.find('DB').find('points_').findall('item')):
            id_ = int(d.find('px').find('id_').text)
            assert (id_ == i)
            val[i]['laser_id'] = id_
            val[i]['rotational_correction'] = float(d.find('px').find('rotCorrection_').text)
            val[i]['vertical_correction'] = float(d.find('px').find('vertCorrection_').text)
            val[i]['distance_far_correction'] = float(d.find('px').find('distCorrection_').text)
            val[i]['distance_correction_x'] = float(d.find('px').find('distCorrectionX_').text)
            val[i]['distance_correction_y'] = float(d.find('px').find('distCorrectionY_').text)
            val[i]['vertical_offset_correction'] = float(d.find('px').find('vertOffsetCorrection_').text)
            val[i]['horizontal_offset_correction'] = float(d.find('px').find('horizOffsetCorrection_').text)
            val[i]['focal_distance'] = float(d.find('px').find('focalDistance_').text)
            val[i]['focal_slope'] = float(d.find('px').find('focalSlope_').text)

        self.laser_calibs = pd.DataFrame(val)

    def __getitem__(self, idx):
        return self.laser_calibs[idx]

    def __len__(self):
        return self.n_lasers


def read_uint8(data, idx):
  return data[idx]


def read_sint8(data, idx):
  val = read_uint8(data, idx)
  return val-256 if val > 127 else val


def read_uint16(data, idx):
  return data[idx] + data[idx+1]*256


def read_sint16(data, idx):
  val = read_uint16(data, idx)
  return val-2**16 if val > 2**15-1 else val


def read_uint32(data, idx):
  return data[idx] + data[idx+1]*256 + data[idx+2]*256*256 + data[idx+3]*256*256*256


class DataPacket(object):
    def __init__(self, timestamp_packet_receive, bytes_stream):
        assert len(bytes_stream) == 1248
        self.timestamp_packet_receive = timestamp_packet_receive
        eth = dpkt.ethernet.Ethernet(bytes_stream)
        self.src = socket.inet_ntoa(eth.data.src)
        self.dst = socket.inet_ntoa(eth.data.dst)
        self.sport = eth.data.data.sport
        self.dport = eth.data.data.dport
        
        self.data_blocks = [DataBlock(eth.data.data.data[idx_block*100: (idx_block+1)*100]) for idx_block in range(12)]
        # microsecond since toh
        self.timestamp = read_uint32(eth.data.data.data, 1200)

        print("DataPacket [{}] src: {}:{} --> dst:{}:{}".format(self.timestamp, self.src, self.sport, self.dst, self.dport))

        self.factory_byte1 = hex(read_uint8(eth.data.data.data, 1204))
        assert self.factory_byte1 == '0x37' or self.factory_byte1 == '0x38' or self.factory_byte1 == '0x39'
        # if self.factory_byte1 == '0x37':
        #     print('    Return Mode: Strongest')
        # elif self.factory_byte1 == '0x38':
        #     print('    Return Mode: Last Return')
        # elif self.factory_byte1 == '0x39':
        #     print('    Return Mode: Dual Return')

        self.factory_byte2 = hex(read_uint8(eth.data.data.data, 1205))
        assert self.factory_byte2 == '0xa1'
        # if self.factory_byte2 == '0xa1':
        #     print('    Product ID: VLS-128')


    def get_pts_info(self):
        res = None
        for i in range(len(self.data_blocks)):
            if res is None:
                res = self.data_blocks[i].get_pts_info()
                res['idx_datablock'] = i
            else:
                temp = self.data_blocks[i].get_pts_info()
                temp['idx_datablock'] = i
                res = pd.concat([res, temp], axis=0, ignore_index=True)

        return res

class DataBlock(object):
    def __init__(self, data):
        # flag
        self.flag1 = hex(data[0])
        assert self.flag1 == '0xff'
        self.flag2 = hex(data[1])
        assert self.flag2 == '0xee' or self.flag2 == '0xdd' or self.flag2 == '0xcc' or self.flag2 == '0xbb'

        # azimuth, degree
        self.azimuth = read_uint16(data, 2) / 100

        # 32 data points
        self.data_points = [DataPoint(data[4+idx_point*3: 4+(idx_point+1)*3]) for idx_point in range(32)]

    def get_pts_info(self):
        if self.flag2 == '0xee':
            laser_offset = 0
        elif self.flag2 == '0xdd':
            laser_offset = 32
        elif self.flag2 == '0xcc':
            laser_offset = 64
        elif self.flag2 == '0xbb':
            laser_offset = 96
        res = []
        for i in range(len(self.data_points)):
            res.append({
                'idx_datapoint': i,
                'idx_laser': laser_offset + i,
                'azimuth': self.azimuth,
                'distance': self.data_points[i].distance,
                'intensity': self.data_points[i].intensity
            })
        res = pd.DataFrame(res)
        return res


class DataPoint(object):
    def __init__(self, data):
        # distance
        distance_res = 0.004
        self.distance = read_uint16(data, 0) * distance_res
        # intensity
        self.intensity = read_uint8(data, 2)


class PositionPacket(object):
    def __init__(self, timestamp_packet_receive, bytes_stream):
        assert len(bytes_stream) == 554
        self.timestamp_packet_receive = timestamp_packet_receive
        eth = dpkt.ethernet.Ethernet(bytes_stream)
        self.src = socket.inet_ntoa(eth.data.src)
        self.dst = socket.inet_ntoa(eth.data.dst)
        self.sport = eth.data.data.sport
        self.dport = eth.data.data.dport

        self.temperature_of_top_board = read_uint8(eth.data.data.data, 187)
        self.temperature_of_bottom_board = read_uint8(eth.data.data.data, 188)
        self.temperature_when_adc_calibration_last_ran = read_uint8(eth.data.data.data, 189)
        self.change_in_temperature_since_last_adc_calibration = read_sint16(eth.data.data.data, 190)
        self.elapsed_seconds_since_last_adc_calibration = read_uint32(eth.data.data.data, 192)
        self.reason_for_the_last_adc_calibration = read_uint8(eth.data.data.data, 196)
        self.bitmask_indicating_current_status_of_adc_calibration = read_uint8(eth.data.data.data, 197)

        self.microsec_since_top_of_the_hour = read_uint32(eth.data.data.data, 198)

        print("PositionPacket [{}] src: {}:{} --> dst:{}:{}".format(self.microsec_since_top_of_the_hour, self.src, self.sport, self.dst, self.dport))

        self.pulse_per_second_status = read_uint8(eth.data.data.data, 202)
        if self.pulse_per_second_status == 0:
            log_YELLOW('    PPS Status: Absent. No PPS detected.')
        elif self.pulse_per_second_status == 1:
            log_YELLOW('    PPS Status: Synchronizing. Synchronizing to PPS.')
        elif self.pulse_per_second_status == 2:
            log_GREEN('    PPS Status: Locked. PPS Locked.')
        else:
            log_RED('    PPS Status: Error. Error.')
            # exit()

        self.thermal_status = read_uint8(eth.data.data.data, 203)
        self.last_shutdown_temperature = read_uint8(eth.data.data.data, 204)
        self.temperature_of_unit_at_power_up = read_uint8(eth.data.data.data, 205)
        self.nmea_sentence = eth.data.data.data[206:334]
        self.nmea_info = NMEAInfo(self.nmea_sentence.decode().split('\r\n')[0])


class NMEAInfo(object):
    def __init__(self, nmea_str):
        feilds = nmea_str.split(',')
        self.format_id = feilds[0]

        self.utc_hour = int(feilds[1][0:2])
        self.utc_minute = int(feilds[1][2:4])
        self.utc_second = int(feilds[1][4:6])
        self.utc_subsecond = int(feilds[1].split('.')[-1])

        self.state = feilds[2]  # A: success, V: fail
        assert self.state == 'A' or self.state == 'V'
        if self.state == 'A':
            log_GREEN('    NMEA success')
        elif self.state == 'V':
            log_RED('    NMEA fail')

        assert feilds[4] == 'N' or feilds[4] == 'S'
        if feilds[4] == 'N':
            self.latitude = int(feilds[3][:2]) + float(feilds[3][2:]) / 60  # degree
        elif feilds[4] == 'S':
            self.latitude = -(int(feilds[3][:2]) + float(feilds[3][2:]) / 60)

        assert feilds[6] == 'E' or feilds[6] == 'W'
        if feilds[6] == 'E':
            self.longitude = int(feilds[5][:3]) + float(feilds[5][3:])/60
        elif feilds[6] == 'W':
            self.longitude = -(int(feilds[5][:3]) + float(feilds[5][3:]) / 60)

        self.velocity = float(feilds[7]) * 1.852  # km/h
        self.heading = float(feilds[8])  # degree

        self.utc_day = int(feilds[9][0:2])
        self.utc_month = int(feilds[9][2:4])
        self.utc_year = int(feilds[9][4:6]) + 2000

        self.magnetic_declination = float(feilds[10])
        self.magnetic_declination_orientation = feilds[11]

        self.mode = feilds[12].split('*')[0]


class VelodyneDecoder(object):
    def __init__(self, config_path, pcap_path, output_path, frame_cut_degree=180):
        print('=' * 100)
        print('Initialization\n')

        self.config_path = config_path
        print('Using calibration file: {}\n'.format(self.config_path))
        self.lidar_config = LidarCalib(config_path)

        assert pcap_path.split('.')[-1] == 'pcap'
        self.pcap_path = pcap_path
        print('Reading packets from: {}\n'.format(self.pcap_path))

        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print('Create output folder: {}'.format(self.output_path))

        self.data_packet = None
        self.data_packet_last = None
        self.position_packet = None
        self.position_packet_last = None
        self.microsecond_since_toh_last_packet = None

        self.f = open(self.pcap_path, 'rb')
        self.reader = enumerate(dpkt.pcap.Reader(self.f))
        self.idx_packet = None

        self.toh = None
        while True:
            # util data packet followed by position packet occur
            try:
                self.idx_packet, (ts, pkg) = next(self.reader)
                print('idx_packet={}, '.format(self.idx_packet), end='')
                if len(pkg) == 1248:
                    self.data_packet = DataPacket(ts, pkg)
                    self.microsecond_since_toh_last_packet = self.data_packet.timestamp
                elif len(pkg) == 554:
                    if self.data_packet is None:
                        continue
                    self.position_packet = PositionPacket(ts, pkg)
                    # utc time
                    toh_utc_time_str = '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(
                        self.position_packet.nmea_info.utc_year, self.position_packet.nmea_info.utc_month,
                        self.position_packet.nmea_info.utc_day,
                        self.position_packet.nmea_info.utc_hour, 0, 0
                    )
                    # unix timestamp
                    self.toh = time.mktime(
                        time.strptime(toh_utc_time_str, '%Y-%m-%d %H:%M:%S')
                    ) - time.timezone
                    # local time
                    toh_local_time_str = time.strftime(
                        '%Y-%m-%d %H:%M:%S', time.localtime(self.toh)
                    )
                    log_BLUE('    toh local_time: {}'.format(toh_local_time_str))
                    log_BLUE('    toh utc_time: {}'.format(toh_utc_time_str))
                    log_BLUE('    toh unix_timestamp: {:.6f}'.format(self.toh))
                    self.microsecond_since_toh_last_packet = self.position_packet.microsec_since_top_of_the_hour
                    break
                else:
                    log_YELLOW('##WARNING: skip other packet, len(bytes_stream) = {}'.format(len(pkg)))

            except Exception as e:
                print('Read all packets done')
                log_YELLOW(repr(e))
                exit()

        self.data_packet_delay_tolerate_thred = -160
        self.position_packet_delay_tolerate_thred = -160


        self.pts_packet = None

        assert frame_cut_degree >= 0 and frame_cut_degree < 360
        self.frame_cut_degree = frame_cut_degree  # begin with frame_cut_degree, clockwise, end up with frame_cut_degree, form a pc frame
        self.pts_frame = None
        self.frame_degrees = np.array([])


    def __del__(self):
        self.f.close()

    def decode_next_packet(self):
        print('='*100)
        try:
            self.idx_packet, (ts, pkg) = next(self.reader)
            print('idx_packet={}, '.format(self.idx_packet), end='')
            if len(pkg) == 1248:
                # backup last data packet
                self.data_packet_last = self.data_packet

                # decode data packet
                self.data_packet = DataPacket(ts, pkg)

                is_delay_packet, time_offset = self.maintain_toh_by_datapacket()
                if is_delay_packet:
                    # delay happened
                    if time_offset > self.data_packet_delay_tolerate_thred:
                        # time delay is small
                        log_YELLOW('    time_delay({}) is small'.format(time_offset))
                        self.cal_pts_in_packet()
                    else:
                        # time_delay is too big, skip this packet
                        log_YELLOW('    time_delay({}) is too big, skip this packet'.format(time_offset))
                        self.data_packet = self.data_packet_last
                else:
                    # no delay happened
                    self.cal_pts_in_packet()

                self.microsecond_since_toh_last_packet = self.data_packet.timestamp

            elif len(pkg) == 554:
                self.position_packet_last = self.position_packet

                # decode position packet
                self.position_packet = PositionPacket(ts, pkg)

                is_delay_packet, time_offset = self.maintain_toh_by_positionpacket()

                if is_delay_packet:
                    if time_offset > self.position_packet_delay_tolerate_thred:
                        # time delay is small
                        log_YELLOW('    time_delay({}) is small'.format(time_offset))
                    else:
                        # time_delay is too big, skip this packet
                        log_YELLOW('    time_delay({}) is too big, skip this packet'.format(time_offset))
                        self.position_packet = self.position_packet_last

                # update self.microsecond_since_toh_last_packet
                self.microsecond_since_toh_last_packet = self.position_packet.microsec_since_top_of_the_hour

            else:
                log_YELLOW('##WARNING: skip other packet, len(bytes_stream) = {}'.format(len(pkg)))

        except Exception as e:
            self.generate_frame(pcd_file_type='pcd')
            print('Read all packets done')
            log_YELLOW(repr(e))
            exit()


    def maintain_toh_by_datapacket(self):
        is_delay_packet = False

        time_offset = self.data_packet.timestamp - self.microsecond_since_toh_last_packet
        if time_offset < 0:
            if time_offset < -3599 * 1e6:
                self.toh += 3600
                # utc time
                toh_utc_time_str = time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.gmtime(self.toh)
                )
                # local time
                toh_local_time_str = time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime(self.toh)
                )
                log_BLUE('    Updated toh local_time: {}'.format(toh_local_time_str))
                log_BLUE('    Updated toh utc_time: {}'.format(toh_utc_time_str))
                log_BLUE('    Updated toh unix_timestamp: {:.6f}'.format(self.toh))
            else:
                is_delay_packet = True

        return is_delay_packet, time_offset



    def maintain_toh_by_positionpacket(self):
        is_delay_packet = False

        time_offset = self.position_packet.microsec_since_top_of_the_hour - self.microsecond_since_toh_last_packet
        if time_offset < 0:
            if time_offset < -3599 * 1e6:
                self.toh += 3600
                # utc time
                toh_utc_time_str = time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.gmtime(self.toh)
                )
                # local time
                toh_local_time_str = time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime(self.toh)
                )
                log_BLUE('    Updated toh local_time: {}'.format(toh_local_time_str))
                log_BLUE('    Updated toh utc_time: {}'.format(toh_utc_time_str))
                log_BLUE('    Updated toh unix_timestamp: {:.6f}'.format(self.toh))
            else:
                is_delay_packet = True

        # from NMEA
        toh_from_nmea = time.mktime(
            time.strptime(
                '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(
                    self.position_packet.nmea_info.utc_year, self.position_packet.nmea_info.utc_month,
                    self.position_packet.nmea_info.utc_day,
                    self.position_packet.nmea_info.utc_hour, 0, 0
                ),
                '%Y-%m-%d %H:%M:%S'
            )
        ) - time.timezone

        # from inside_clock
        toh_inside_clock = self.toh

        if toh_from_nmea == toh_inside_clock:
            log_GREEN('    toh_from_nmea = {}, toh_inside_clock = {}'.format(toh_from_nmea, toh_inside_clock))
        else:
            log_YELLOW('    toh_from_nmea = {}, toh_inside_clock = {}'.format(toh_from_nmea, toh_inside_clock))
            key = input('Continue?(y/n)[y]')
            if key == 'n' or key == 'N':
                exit()

        return is_delay_packet, time_offset


    def cal_pts_in_packet(self):
        self.pts_packet = self.data_packet.get_pts_info()
        self.pts_packet['idx_sequence'] = np.int64(np.floor(self.pts_packet['idx_datablock'].values / 4))
        self.pts_packet['idx_firinggroup'] = np.int64(np.floor(self.pts_packet['idx_laser'].values / 8))

        # microsecond since toh
        # Alpha Prime
        time_per_firing_group = 2.66666
        time_rp0 = 5.33333
        time_rp1 = 5.33333
        time_rp2_avg = 3.882 / 2
        time_per_firing_sequence_avg = time_per_firing_group * 8 + time_rp0 \
                                       + time_per_firing_group * 8 + time_rp1 + time_rp2_avg
        self.pts_packet['time_offset'] = (time_per_firing_sequence_avg * self.pts_packet['idx_sequence'].values) \
                                         + (time_per_firing_group * self.pts_packet['idx_firinggroup'].values) \
                                         + (self.pts_packet['idx_firinggroup'].values >= 8) * time_rp0 - 7
        self.pts_packet['microsec_since_toh'] = self.data_packet.timestamp + self.pts_packet['time_offset'].values

        # get unix timestamp
        self.pts_packet['unix_timestamp'] = self.pts_packet['microsec_since_toh'].values / 1e6 + self.toh

        # precision azi
        if self.data_packet.data_blocks[0].azimuth < self.data_packet_last.data_blocks[0].azimuth:
            azimuth_gap = 360 + self.data_packet.data_blocks[0].azimuth - self.data_packet_last.data_blocks[0].azimuth
        else:
            azimuth_gap = self.data_packet.data_blocks[0].azimuth - self.data_packet_last.data_blocks[0].azimuth

        if self.data_packet.timestamp < self.data_packet_last.timestamp:
            time_gap = 3600000000 + self.data_packet.timestamp - self.data_packet_last.timestamp
        else:
            time_gap = self.data_packet.timestamp - self.data_packet_last.timestamp

        azimuth_rate = azimuth_gap / time_gap  # degree/microsecond

        self.pts_packet['azi_offset'] = self.lidar_config.laser_calibs.iloc[self.pts_packet['idx_laser'].values]['rotational_correction'].values

        self.pts_packet['azi'] = self.pts_packet['azimuth'].values + azimuth_rate * (
                (time_per_firing_group * self.pts_packet['idx_firinggroup'].values)
                + (self.pts_packet['idx_firinggroup'].values >= 8) * time_rp0 - 7
        ) - self.pts_packet['azi_offset'].values

        self.pts_packet['ele'] = self.lidar_config.laser_calibs.iloc[self.pts_packet['idx_laser'].values]['vertical_correction'].values

        # calculate xyz
        self.pts_packet['x'] = self.pts_packet['distance'].values * np.cos(self.pts_packet['ele'].values/180*np.pi) * np.sin(self.pts_packet['azi'].values/180*np.pi)
        self.pts_packet['y'] = self.pts_packet['distance'].values * np.cos(self.pts_packet['ele'].values/180*np.pi) * np.cos(self.pts_packet['azi'].values/180*np.pi)
        self.pts_packet['z'] = self.pts_packet['distance'].values * np.sin(self.pts_packet['ele'].values/180*np.pi)

        # delete pts distance==0
        self.pts_packet.drop(np.nonzero((self.pts_packet['distance'] == 0).values)[0], inplace=True)

        # maintain frame_degrees
        self.frame_degrees = np.concatenate((self.frame_degrees, np.unique(self.pts_packet['azimuth'].values)), axis=0)

        # maintain pts_frame
        if self.pts_frame is None:
            self.pts_frame = self.pts_packet
        else:
            self.pts_frame = pd.concat([self.pts_frame, self.pts_packet], axis=0, ignore_index=True)

    def judge_jump_cut_degree(self):
        protect_width = 3
        res = np.where(self.frame_degrees < self.frame_cut_degree)[0]
        if len(res) <= protect_width:
            return False
        else:
            idx_start = res[protect_width]
            return self.frame_degrees[idx_start:].max() >= self.frame_cut_degree

    def generate_frame(self, pcd_file_type='pcd'):
        # name the file according to the pts' unix_timestamp at 0 degree
        pcd_filename = '{:.6f}.{}'.format(
            self.pts_frame['unix_timestamp'].iloc[np.abs(self.pts_frame['azi'].values).argmin()],
            pcd_file_type
        )
        pcd_path = os.path.join(self.output_path, pcd_filename)

        # x y z intensity idx_laser unix_timestamp
        if pcd_file_type == 'npz':
            np.savez(
                pcd_path,
                x=self.pts_frame['x'].values.astype('float32'),
                y=self.pts_frame['y'].values.astype('float32'),
                z=self.pts_frame['z'].values.astype('float32'),
                intensity=self.pts_frame['intensity'].values.astype('uint8'),
                idx_laser=self.pts_frame['idx_laser'].values.astype('uint8'),
                unix_timestamp=self.pts_frame['unix_timestamp'].values.astype('float64')
            )
        elif pcd_file_type == 'bin':
            pass
        elif pcd_file_type == 'pcd':
            pcd = pd.DataFrame({
                'x': self.pts_frame['x'].values.astype('float32'),
                'y': self.pts_frame['y'].values.astype('float32'),
                'z': self.pts_frame['z'].values.astype('float32'),
                'intensity': self.pts_frame['intensity'].values.astype('uint8'),
                'idx_laser': self.pts_frame['idx_laser'].values.astype('uint8'),
                'unix_timestamp': self.pts_frame['unix_timestamp'].values.astype('float64'),
            })

            pcd.to_csv(pcd_path, sep=' ', index=False, header=False)
            with open(pcd_path, 'r') as f_pcd:
                lines = f_pcd.readlines()

            with open(pcd_path, 'w') as f_pcd:
                f_pcd.write('VERSION .7\n')
                f_pcd.write('FIELDS')
                for col in pcd.columns.values:
                    f_pcd.write(' {}'.format(col))
                f_pcd.write('\n')
                f_pcd.write('SIZE 4 4 4 1 1 8\n')
                f_pcd.write('TYPE F F F U U F\n')
                f_pcd.write('COUNT 1 1 1 1 1 1\n')
                f_pcd.write('WIDTH {}\n'.format(len(pcd)))
                f_pcd.write('HEIGHT 1\n')
                f_pcd.write('VIEWPOINT 0 0 0 1 0 0 0\n')
                f_pcd.write('POINTS {}\n'.format(len(pcd)))
                f_pcd.write('DATA ascii\n')
                f_pcd.writelines(lines)

        log_GREEN('    Generate {},save to {}'.format(pcd_filename, pcd_path))

        # new frame init
        self.pts_frame = None
        self.frame_degrees = np.array([])


def test():
    ts_str = '1671267290.996'

if __name__ == '__main__':
    test()
