import os
import re
import json

import numpy as np

from utils import *

class RecordData_NormalMode(object):
    def __init__(self, path):
        self.root = path
        self.folder = os.path.basename(self.root)
        tmp = self.folder.split('_')
        self.time = tmp[0]+'_'+tmp[1]
        self.mode = tmp[2]
        assert self.mode != 'mixmode'
        self.n_frames = int(tmp[3])

        self.bin_files = []

        for file in os.listdir(self.root):
            if re.match('\w*.mmwave.json', file):
                self.mmwave_json_file = file

            if re.match('\w*.triggertime.txt', file):
                with open(os.path.join(self.root, file), 'r') as f:
                    contents = f.readlines()
                assert len(contents) == 1
                self.triggertime = contents[0].replace(' ', '').replace('\n', '')

            if re.match('\w*.bin', file):
                self.bin_files.append(file)

        self.n_slices = int(len(self.bin_files) / 4 / 2)

        self.mode_infos = ModeInfo_NormalMode(os.path.join(self.root, self.mmwave_json_file))

    def divide_frame(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        n_frames_in_slices = np.zeros((self.n_slices,), dtype='int')
        for i in range(self.n_slices):
            log_BLUE('({}/{}) id_slice={}'.format(i+1, self.n_slices, i))
            master_idx_bin_path = os.path.join(self.root, 'master_{:>04d}_idx.bin'.format(i))
            n_frames_in_slices[i] = get_n_frames_in_slice(master_idx_bin_path)
            for j in range(n_frames_in_slices[i]):
                id_frame_in_slice = j
                n_frames_in_slice = n_frames_in_slices[i]
                id_frame_global = id_frame_in_slice + n_frames_in_slices[:i].sum()
                n_frames_global = self.mode_infos.numFrames

                log('  ({:>3d}/{:>3d}) id_frame_in_slice={:>3d}, ({:>3d}/{:>3d}) id_frame_in_slice={:>3d}'.format(
                    id_frame_in_slice + 1, n_frames_in_slice, id_frame_in_slice,
                    id_frame_global + 1, n_frames_global, id_frame_global
                ))

                # get current frame's timestamp
                msec_add = self.mode_infos.framePeriodicity_msec * id_frame_global
                timestamp = timestamp_add(self.triggertime, msec_add)

                # judge: whether current frame is already generated?
                output_path = os.path.join(output_folder, '{}.npz'.format(timestamp))
                if os.path.exists(output_path):
                    log_YELLOW('  already generated {}'.format(output_path))
                    continue

                # read adc data
                master_data_bin_path = os.path.join(self.root, 'master_{:>04d}_data.bin'.format(i))
                slave1_data_bin_path = os.path.join(self.root, 'slave1_{:>04d}_data.bin'.format(i))
                slave2_data_bin_path = os.path.join(self.root, 'slave2_{:>04d}_data.bin'.format(i))
                slave3_data_bin_path = os.path.join(self.root, 'slave3_{:>04d}_data.bin'.format(i))

                data_RXchain_master_real, data_RXchain_master_imag = read_data_bin(
                    master_data_bin_path, id_frame_in_slice, self.mode_infos.numAdcSamples,
                    self.mode_infos.numChirps, self.mode_infos.numLoops, self.mode_infos.numRXPerDevice
                )
                data_RXchain_slave1_real, data_RXchain_slave1_imag = read_data_bin(
                    slave1_data_bin_path, id_frame_in_slice, self.mode_infos.numAdcSamples,
                    self.mode_infos.numChirps, self.mode_infos.numLoops, self.mode_infos.numRXPerDevice
                )
                data_RXchain_slave2_real, data_RXchain_slave2_imag = read_data_bin(
                    slave2_data_bin_path, id_frame_in_slice, self.mode_infos.numAdcSamples,
                    self.mode_infos.numChirps, self.mode_infos.numLoops, self.mode_infos.numRXPerDevice
                )
                data_RXchain_slave3_real, data_RXchain_slave3_imag = read_data_bin(
                    slave3_data_bin_path, id_frame_in_slice, self.mode_infos.numAdcSamples,
                    self.mode_infos.numChirps, self.mode_infos.numLoops, self.mode_infos.numRXPerDevice
                )

                data_RXchain_real = np.concatenate(
                    (data_RXchain_master_real, data_RXchain_slave1_real, data_RXchain_slave2_real, data_RXchain_slave3_real),
                    axis=2
                )
                data_RXchain_imag = np.concatenate(
                    (data_RXchain_master_imag, data_RXchain_slave1_imag, data_RXchain_slave2_imag, data_RXchain_slave3_imag),
                    axis=2
                )

                # save every frame: adcdata + mode_infos
                output_path = os.path.join(output_folder, '{}.npz'.format(timestamp))
                mode_infos_dict = dict(
                    [(attr, self.mode_infos.__getattribute__(attr))
                     for attr in dir(self.mode_infos) if not attr.startswith("__")]
                )
                mode_infos_dict['name'] = self.mode
                np.savez(
                    output_path,
                    data_real=data_RXchain_real,
                    data_imag=data_RXchain_imag,
                    mode_infos=mode_infos_dict
                )

                log_GREEN('  save as {}'.format(output_path))


class ModeInfo_NormalMode(object):
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.startFreqConst_GHz = data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']['startFreqConst_GHz']
        self.idleTimeConst_usec = data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']['idleTimeConst_usec']
        self.adcStartTimeConst_usec = data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']['adcStartTimeConst_usec']
        self.rampEndTime_usec = data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']['rampEndTime_usec']
        self.freqSlopeConst_MHz_usec = data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']['freqSlopeConst_MHz_usec']
        self.numAdcSamples = data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']['numAdcSamples']
        self.digOutSampleRate_ksps = data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][0]['rlProfileCfg_t']['digOutSampleRate']

        self.chirpEndIdx = data['mmWaveDevices'][0]['rfConfig']['rlFrameCfg_t']['chirpEndIdx']
        self.chirpStartIdx = data['mmWaveDevices'][0]['rfConfig']['rlFrameCfg_t']['chirpStartIdx']
        self.numLoops = data['mmWaveDevices'][0]['rfConfig']['rlFrameCfg_t']['numLoops']
        self.numFrames = data['mmWaveDevices'][0]['rfConfig']['rlFrameCfg_t']['numFrames']
        self.framePeriodicity_msec = data['mmWaveDevices'][0]['rfConfig']['rlFrameCfg_t']['framePeriodicity_msec']

        self.numChirps = len(data['mmWaveDevices'][0]['rfConfig']['rlChirps'])
        self.light_speed_m_sec = 3e8
        self.numDevices = len(data['mmWaveDevices'])
        # AWR2243P
        self.numRXPerDevice = 4
        self.numTXPerDevice = 3

        self.bandwidth_MHz = self.numAdcSamples / (self.digOutSampleRate_ksps * 1e3) * (self.freqSlopeConst_MHz_usec * 1e6 / 1e-6) / 1e6
        self.bandwidth_MHz_valid = (self.rampEndTime_usec * 1e-6) * (self.freqSlopeConst_MHz_usec * 1e6 / 1e-6) / 1e6
        self.Tchirp_usec = self.idleTimeConst_usec + self.rampEndTime_usec
        self.Tloop_usec = self.Tchirp_usec * self.numChirps
        self.freq_center_GHz = self.startFreqConst_GHz + (self.adcStartTimeConst_usec + self.numAdcSamples / 2 / self.digOutSampleRate_ksps * 1e3) * self.freqSlopeConst_MHz_usec / 1e3
        self.lambda_center_mm = self.light_speed_m_sec / (self.freq_center_GHz * 1e9) * 1e3
        self.Tframe_msec = self.Tloop_usec * self.numLoops / 1e3

        self.range_max_m = (self.digOutSampleRate_ksps * 1e3) * self.light_speed_m_sec / 2 / (self.freqSlopeConst_MHz_usec * 1e6 / 1e-6)
        self.range_res_m = self.light_speed_m_sec / 2 / (self.bandwidth_MHz * 1e6)
        self.velocity_max_m_sec = (self.lambda_center_mm * 1e-3) / 4 / (self.Tloop_usec * 1e-6)
        self.velocity_res_m_sec = (self.lambda_center_mm * 1e-3) / 2 / (self.Tframe_msec * 1e-3)
        self.fps_Hz = 1 / (self.framePeriodicity_msec * 1e-3)


def read_data_bin(data_bin_path, id_frame_in_slice, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice):
    num_IQSample = 2
    Expected_Num_SamplesPerFrame = numSamplePerChirp * numChirpPerLoop * numLoops * numRXPerDevice * num_IQSample
    numBytesPerSample = 2  # every sample is int16
    num_bytes_skip = id_frame_in_slice * Expected_Num_SamplesPerFrame * numBytesPerSample
    with open(data_bin_path, 'rb') as f:
        f.seek(num_bytes_skip, 0)
        adc_data = f.read(Expected_Num_SamplesPerFrame * numBytesPerSample)

    cache_path = './tmp_read_data_bin.bin'
    np.array(adc_data).tofile(cache_path)
    adc_data = np.fromfile(cache_path, dtype=np.int16)
    os.remove(cache_path)

    real = adc_data[::2]
    real = real.reshape((numRXPerDevice, numSamplePerChirp, numChirpPerLoop, numLoops), order='F')
    real = real.transpose(1, 3, 0, 2)

    imag = adc_data[1::2]
    imag = imag.reshape((numRXPerDevice, numSamplePerChirp, numChirpPerLoop, numLoops), order='F')
    imag = imag.transpose(1, 3, 0, 2)

    return real, imag


def get_n_frames_in_slice(master_idx_bin_path):
    with open(master_idx_bin_path, 'rb') as f:
        f.seek(12, 0)
        tmp = f.read(4)
        n_frames_valid = tmp[0] + tmp[1] * 256 + tmp[2] * 256 * 256 + tmp[3] * 256 * 256 * 256
    return n_frames_valid


def timestamp_add(ts_str, msec_add):
    n_behind_dot = len(ts_str) - ts_str.find('.') - 1
    res = float(ts_str) + msec_add * 1e-3
    res_str = '{}'.format(round(res, n_behind_dot))
    return res_str


class RecordData_MixMode(object):
    def __init__(self, path):
        self.root = path
        self.folder = os.path.basename(self.root)
        tmp = self.folder.split('_')
        self.time = tmp[0]+'_'+tmp[1]
        self.mode = tmp[2]
        assert self.mode == 'mixmode'
        self.n_frames = int(tmp[3])

        self.bin_files = []

        for file in os.listdir(self.root):
            if re.match('\w*.mmwave.json', file):
                self.mmwave_json_file = file

            if re.match('\w*.triggertime.txt', file):
                with open(os.path.join(self.root, file), 'r') as f:
                    contents = f.readlines()
                assert len(contents) == 1
                self.triggertime = contents[0].replace(' ', '').replace('\n', '')

            if re.match('\w*.bin', file):
                self.bin_files.append(file)

        self.n_slices = int(len(self.bin_files) / 4 / 2)

        self.mode_infos = ModeInfo_MixMode(os.path.join(self.root, self.mmwave_json_file))

    def divide_frame(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        n_subframes_in_slices = np.zeros((self.n_slices,), dtype='int')
        n_frames_in_slices = np.zeros((self.n_slices,), dtype='int')
        for i in range(self.n_slices):
            log_BLUE('({}/{}) id_slice={}'.format(i+1, self.n_slices, i))
            master_idx_bin_path = os.path.join(self.root, 'master_{:>04d}_idx.bin'.format(i))
            n_subframes_in_slices[i] = get_n_frames_in_slice(master_idx_bin_path)
            n_frames_in_slices[i] = n_subframes_in_slices[i] // self.mode_infos.numOfSubFrames
            for j in range(n_frames_in_slices[i]):
                id_frame_in_slice = j
                n_frames_in_slice = n_frames_in_slices[i]
                id_frame_global = id_frame_in_slice + n_frames_in_slices[:i].sum()
                n_frames_global = self.mode_infos.numFrames

                log('  ({:>3d}/{:>3d}) id_frame_in_slice={:>3d}, ({:>3d}/{:>3d}) id_frame_in_slice={:>3d}'.format(
                    id_frame_in_slice + 1, n_frames_in_slice, id_frame_in_slice,
                    id_frame_global + 1, n_frames_global, id_frame_global
                ))

                # get current frame's timestamp
                msec_add = self.mode_infos.framePeriodicity_msec * id_frame_global
                timestamp = timestamp_add(self.triggertime, msec_add)

                # judge: whether current frame is already generated?
                output_path = os.path.join(output_folder, '{}.npz'.format(timestamp))
                if os.path.exists(output_path):
                    log_YELLOW('  already generated {}'.format(output_path))
                    continue

                # read adc data
                master_data_bin_path = os.path.join(self.root, 'master_{:>04d}_data.bin'.format(i))
                slave1_data_bin_path = os.path.join(self.root, 'slave1_{:>04d}_data.bin'.format(i))
                slave2_data_bin_path = os.path.join(self.root, 'slave2_{:>04d}_data.bin'.format(i))
                slave3_data_bin_path = os.path.join(self.root, 'slave3_{:>04d}_data.bin'.format(i))

                data_RXchain_real = []
                data_RXchain_imag = []
                for k in range(self.mode_infos.numOfSubFrames):
                    id_subframe_in_slice = id_frame_in_slice * self.mode_infos.numOfSubFrames + k

                    data_RXchain_master_real, data_RXchain_master_imag = read_data_bin(
                        master_data_bin_path, id_subframe_in_slice, self.mode_infos.numAdcSamples[k],
                        self.mode_infos.numChirps[k], self.mode_infos.numLoops[k], self.mode_infos.numRXPerDevice
                    )
                    data_RXchain_slave1_real, data_RXchain_slave1_imag = read_data_bin(
                        slave1_data_bin_path, id_subframe_in_slice, self.mode_infos.numAdcSamples[k],
                        self.mode_infos.numChirps[k], self.mode_infos.numLoops[k], self.mode_infos.numRXPerDevice
                    )
                    data_RXchain_slave2_real, data_RXchain_slave2_imag = read_data_bin(
                        slave2_data_bin_path, id_subframe_in_slice, self.mode_infos.numAdcSamples[k],
                        self.mode_infos.numChirps[k], self.mode_infos.numLoops[k], self.mode_infos.numRXPerDevice
                    )
                    data_RXchain_slave3_real, data_RXchain_slave3_imag = read_data_bin(
                        slave3_data_bin_path, id_subframe_in_slice, self.mode_infos.numAdcSamples[k],
                        self.mode_infos.numChirps[k], self.mode_infos.numLoops[k], self.mode_infos.numRXPerDevice
                    )

                    data_RXchain_real.append(
                        np.concatenate(
                            (data_RXchain_master_real, data_RXchain_slave1_real, data_RXchain_slave2_real,
                             data_RXchain_slave3_real),
                            axis=2
                        )
                    )

                    data_RXchain_imag.append(
                        np.concatenate(
                            (data_RXchain_master_imag, data_RXchain_slave1_imag, data_RXchain_slave2_imag,
                             data_RXchain_slave3_imag),
                            axis=2
                        )
                    )

                # save every frame: adcdata + mode_infos
                output_path = os.path.join(output_folder, '{}.npz'.format(timestamp))
                mode_infos_dict = dict(
                    [(attr, self.mode_infos.__getattribute__(attr))
                     for attr in dir(self.mode_infos) if not attr.startswith("__")]
                )
                mode_infos_dict['name'] = self.mode
                np.savez(
                    output_path,
                    data_real=data_RXchain_real,
                    data_imag=data_RXchain_imag,
                    mode_infos=mode_infos_dict
                )

                log_GREEN('  save as {}'.format(output_path))


class ModeInfo_MixMode(object):
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.numOfSubFrames = data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['numOfSubFrames']
        n_mixmode = self.numOfSubFrames

        self.startFreqConst_GHz = [data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][i]['rlProfileCfg_t']['startFreqConst_GHz'] for i in range(n_mixmode)]
        self.idleTimeConst_usec = [data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][i]['rlProfileCfg_t']['idleTimeConst_usec'] for i in range(n_mixmode)]
        self.adcStartTimeConst_usec = [data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][i]['rlProfileCfg_t']['adcStartTimeConst_usec'] for i in range(n_mixmode)]
        self.rampEndTime_usec = [data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][i]['rlProfileCfg_t']['rampEndTime_usec'] for i in range(n_mixmode)]
        self.freqSlopeConst_MHz_usec = [data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][i]['rlProfileCfg_t']['freqSlopeConst_MHz_usec'] for i in range(n_mixmode)]
        self.numAdcSamples = [data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][i]['rlProfileCfg_t']['numAdcSamples'] for i in range(n_mixmode)]
        self.digOutSampleRate_ksps = [data['mmWaveDevices'][0]['rfConfig']['rlProfiles'][i]['rlProfileCfg_t']['digOutSampleRate'] for i in range(n_mixmode)]

        self.forceProfileIdx = [data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['subFrameCfg'][i]['rlSubFrameCfg_t']['forceProfileIdx'] for i in range(n_mixmode)]
        self.chirpStartIdx = [data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['subFrameCfg'][i]['rlSubFrameCfg_t']['chirpStartIdx'] for i in range(n_mixmode)]
        self.numOfChirps = [data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['subFrameCfg'][i]['rlSubFrameCfg_t']['numOfChirps'] for i in range(n_mixmode)]
        self.numLoops = [data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['subFrameCfg'][i]['rlSubFrameCfg_t']['numLoops'] for i in range(n_mixmode)]
        self.burstPeriodicity_msec = [data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['subFrameCfg'][i]['rlSubFrameCfg_t']['burstPeriodicity_msec'] for i in range(n_mixmode)]
        self.numOfBurst = [data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['subFrameCfg'][i]['rlSubFrameCfg_t']['numOfBurst'] for i in range(n_mixmode)]
        self.numOfBurstLoops = [data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['subFrameCfg'][i]['rlSubFrameCfg_t']['numOfBurstLoops'] for i in range(n_mixmode)]
        self.subFramePeriodicity_msec = [data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['subFrameCfg'][i]['rlSubFrameCfg_t']['subFramePeriodicity_msec'] for i in range(n_mixmode)]

        self.numFrames = data['mmWaveDevices'][0]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']['numFrames']

        self.light_speed_m_sec = 3e8
        self.numDevices = len(data['mmWaveDevices'])
        self.numChirps = self.numOfChirps
        # AWR2243P
        self.numRXPerDevice = 4
        self.numTXPerDevice = 3

        self.bandwidth_MHz = [self.numAdcSamples[i] / (self.digOutSampleRate_ksps[i] * 1e3) * (self.freqSlopeConst_MHz_usec[i] * 1e6 / 1e-6) / 1e6 for i in range(n_mixmode)]
        self.bandwidth_MHz_valid = [(self.rampEndTime_usec[i] * 1e-6) * (self.freqSlopeConst_MHz_usec[i] * 1e6 / 1e-6) / 1e6 for i in range(n_mixmode)]
        self.Tchirp_usec = [self.idleTimeConst_usec[i] + self.rampEndTime_usec[i] for i in range(n_mixmode)]
        self.Tloop_usec = [self.Tchirp_usec[i] * self.numChirps[i] for i in range(n_mixmode)]
        self.freq_center_GHz = [self.startFreqConst_GHz[i] + (self.adcStartTimeConst_usec[i] + self.numAdcSamples[i] / 2 / self.digOutSampleRate_ksps[i] * 1e3) * self.freqSlopeConst_MHz_usec[i] / 1e3 for i in range(n_mixmode)]
        self.lambda_center_mm = [self.light_speed_m_sec / (self.freq_center_GHz[i] * 1e9) * 1e3 for i in range(n_mixmode)]
        self.Tframe_msec = [self.Tloop_usec[i] * self.numLoops[i] / 1e3 for i in range(n_mixmode)]

        self.range_max_m = [(self.digOutSampleRate_ksps[i] * 1e3) * self.light_speed_m_sec / 2 / (self.freqSlopeConst_MHz_usec[i] * 1e6 / 1e-6) for i in range(n_mixmode)]
        self.range_res_m = [self.light_speed_m_sec / 2 / (self.bandwidth_MHz[i] * 1e6) for i in range(n_mixmode)]
        self.velocity_max_m_sec = [(self.lambda_center_mm[i] * 1e-3) / 4 / (self.Tloop_usec[i] * 1e-6) for i in range(n_mixmode)]
        self.velocity_res_m_sec = [(self.lambda_center_mm[i] * 1e-3) / 2 / (self.Tframe_msec[i] * 1e-3) for i in range(n_mixmode)]
        self.framePeriodicity_msec = 0
        for i in range(n_mixmode):
            self.framePeriodicity_msec += self.subFramePeriodicity_msec[i]
        self.fps_Hz = 1 / (self.framePeriodicity_msec * 1e-3)


def test_divide_frame():
    root = 'F:\\20221217\\TIRadar'
    output_root = 'F:\\20221217_process\\TIRadar'
    folder = '20221217_165447_mode1_682'

    record_data = RecordData_NormalMode(os.path.join(root, folder))
    output_folder = os.path.join(output_root, record_data.folder)
    record_data.divide_frame(output_folder)

    print('done')

def test_read():
    root = 'F:\\20221217_process\\TIRadar'
    folder = '20221217_165447_mode1_682'
    for file in os.listdir(os.path.join(root, folder)):
        file_path = os.path.join(root, folder, file)
        print('read {}'.format(file_path))

        data = np.load(file_path, allow_pickle=True)

        data_imag = data['data_imag']
        data_real = data['data_real']
        mode_infos = data['mode_infos'][()]

        print('done')

def test_divide_frame_mixmode():
    root = 'F:\\20221217\\TIRadar'
    output_root = 'F:\\20221217_process\\TIRadar'
    folder = '20221217_170608_mixmode_20'

    record_data = RecordData_MixMode(os.path.join(root, folder))
    output_folder = os.path.join(output_root, record_data.folder)
    record_data.divide_frame(output_folder)

    print('done')

def test_read_mixmode():
    root = 'F:\\20221217_process\\TIRadar'
    folder = '20221217_170608_mixmode_20'
    for file in os.listdir(os.path.join(root, folder)):
        file_path = os.path.join(root, folder, file)
        print('read {}'.format(file_path))

        data = np.load(file_path, allow_pickle=True)

        data_imag = list(data['data_imag'])
        data_real = list(data['data_real'])
        mode_infos = data['mode_infos'][()]

        print('done')

if __name__ == '__main__':
    # test_divide_frame()
    # test_read()

    # test_divide_frame_mixmode()
    test_read_mixmode()

