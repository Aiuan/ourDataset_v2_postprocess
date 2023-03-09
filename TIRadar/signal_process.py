import time
import numpy as np

class NormalModeProcess(object):
    def __init__(self, infos, data_real, data_imag, calib, apply_vmax_extend=True):
        self.infos = infos

        # data_real.shape[2]=data_imag.shape[2]=16: ordered by dev[0,1,2,3]_rx[0,1,2,3],
        # data_real.shape[3]=data_imag.shape[3]=12: ordered by self.tx_id_transfer_order
        self.data_raw = data_real + data_imag * 1j
        self.rangeFFT_size = self.data_raw.shape[0]
        self.dopplerFFT_size = self.data_raw.shape[1]
        self.azimuthFFT_size = 256
        self.elevationFFT_size = 256

        self.TI_Cascade_Antenna_DesignFreq_GHz = 76.8
        # doa_unitDis = 0.5 * (self.infos['startFreqConst_GHz'] + 256 / self.infos['digOutSampleRate_ksps']*self.infos['freqSlopeConst_MHz_usec']/2) / TI_Cascade_Antenna_DesignFreq_GHz
        self.doa_unitDis = 0.5 * self.infos['freq_center_GHz'] / self.TI_Cascade_Antenna_DesignFreq_GHz

        self.range_bins = np.arange(self.rangeFFT_size) * self.infos['range_res_m']
        self.doppler_bins = np.arange(-self.dopplerFFT_size/2, self.dopplerFFT_size/2) * self.infos['velocity_res_m_sec']
        self.azimuth_bins = np.arange(-self.azimuthFFT_size, self.azimuthFFT_size, 2) * np.pi / self.azimuthFFT_size
        self.elevation_bins = np.arange(-self.elevationFFT_size, self.elevationFFT_size, 2) * np.pi / self.elevationFFT_size

        self.calib = calib

        self.num_rx = self.infos['numRXPerDevice'] * self.infos['numDevices']
        self.num_tx = self.infos['numTXPerDevice'] * self.infos['numDevices']

        self.rx_id = np.arange(self.num_rx)
        self.rx_id_onboard = np.array([13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8]) - 1
        self.rx_position_azimuth = np.array([11, 12, 13, 14, 50, 51, 52, 53, 46, 47, 48, 49, 0, 1, 2, 3])
        self.rx_position_elevation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.tx_id_transfer_order = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) - 1
        self.tx_id = np.arange(self.num_tx)
        self.tx_id_onboard = np.array([12, 11, 10, 3, 2, 1, 9, 8, 7, 6, 5, 4]) - 1
        self.tx_position_azimuth = np.array([11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0])
        self.tx_position_elevation = np.array([6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.apply_vmax_extend = apply_vmax_extend
        if self.apply_vmax_extend:
            self.min_dis_apply_vmax_extend = 10

        self.pcd = None
        self.heatmapBEV = None
        self.heatmap4D = None

        self.scale_factor = np.array([0.2500, 0.1250, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039, 0.0020])

    def __timer__(self, fn):
        start_time = time.time()
        fn()
        end_time = time.time()
        print('{}() function consume: {:.3f} s'.format(fn.__name__, end_time - start_time))

    def run(self, generate_pcd, generate_heatmapBEV, generate_heatmap4D):
        self.generate_pcd = generate_pcd
        self.generate_heatmapBEV = generate_heatmapBEV
        self.generate_heatmap4D = generate_heatmap4D

        if self.generate_pcd or self.generate_heatmapBEV or self.generate_heatmap4D:
            self.__timer__(self.__calibrate_rawdata__)

            self.__timer__(self.__re_order__)

            self.__timer__(self.__calculate_antenna_array__)

            self.__timer__(self.__rangeFFT__)

            self.__timer__(self.__dopplerFFT__)

            if self.generate_pcd:
                self.__timer__(self.__rdm_cfar__)

                if self.res_rdm_cfar is not None:
                    self.__timer__(self.__doa__)

                    if self.res_doa is not None:
                        self.__timer__(self.__generate_pcd__)

            if self.generate_heatmapBEV:
                self.__timer__(self.__generate_heatmapBEV__)

            if self.generate_heatmap4D:
                self.__timer__(self.__generate_heatmap4D__)

    def get_range_bins(self):
        return self.range_bins

    def get_doppler_bins(self):
        return self.doppler_bins

    def get_azimuth_bins(self, unit_degree=True):
        if unit_degree:
            return np.arcsin(self.azimuth_bins / 2 / np.pi / self.doa_unitDis) / np.pi * 180
        else:
            return np.arcsin(self.azimuth_bins / 2 / np.pi / self.doa_unitDis)

    def get_elevation_bins(self, unit_degree=True):
        if unit_degree:
            return np.arcsin(self.elevation_bins / 2 / np.pi / self.doa_unitDis) / np.pi * 180
        else:
            return np.arcsin(self.elevation_bins / 2 / np.pi / self.doa_unitDis)

    def get_pcd(self):
        return self.pcd

    def get_heatmapBEV(self):
        return self.heatmapBEV

    def get_heatmap4D(self):
        return self.heatmap4D

    def __calibrate_rawdata__(self):
        adc_sample_rate = self.infos['digOutSampleRate_ksps'] * 1e3
        chirp_slope = self.infos['freqSlopeConst_MHz_usec'] * 1e6 / 1e-6
        num_sample = self.infos['numAdcSamples']
        num_loop = self.infos['numLoops']

        range_mat = self.calib['calibResult']['RangeMat']
        fs_calib = self.calib['params']['Sampling_Rate_sps']
        slope_calib = self.calib['params']['Slope_MHzperus'] * 1e6 / 1e-6
        calibration_interp = 5
        peak_val_mat = self.calib['calibResult']['PeakValMat']
        phase_calib_only = 1

        tx_id_ref = self.tx_id_transfer_order[0]

        # construct the frequency compensation matrix
        freq_calib = 2 * np.pi * (
                (range_mat[self.tx_id_transfer_order, :] - range_mat[tx_id_ref, 0])
                * fs_calib / adc_sample_rate * chirp_slope / slope_calib
        ) / (num_sample * calibration_interp)
        correction_vec = np.exp(
            1j * np.arange(num_sample).reshape((-1, 1, 1)) * np.expand_dims(freq_calib.T, axis=0)
        ).conj()
        freq_correction_mat = np.expand_dims(correction_vec, axis=1)

        # construct the phase compensation matrix
        phase_calib = peak_val_mat[tx_id_ref, 0] / peak_val_mat[self.tx_id_transfer_order, :]
        # remove amplitude calibration
        if phase_calib_only == 1:
            phase_calib = phase_calib / np.abs(phase_calib)
        phase_correction_mat = np.expand_dims(np.expand_dims(phase_calib.T, axis=0), axis=0)

        self.data_calib = self.data_raw * freq_correction_mat * phase_correction_mat

    def __re_order__(self):
        # data_reordered.shape[2]=16: ordered by self.rx_id_onboard
        # data_reordered.shape[3]=12: ordered by self.tx_id_transfer_order
        self.data_reordered = self.data_calib[:, :, self.rx_id_onboard, :]

    def __calculate_antenna_array__(self):
        self.virtual_array_azimuth = np.tile(self.tx_position_azimuth[self.tx_id_transfer_order], (self.num_rx, 1)) + \
                                        np.tile(self.rx_position_azimuth[self.rx_id_onboard], (self.num_tx, 1)).T
        self.virtual_array_elevation = np.tile(self.tx_position_elevation[self.tx_id_transfer_order], (self.num_rx, 1)) + \
                                          np.tile(self.rx_position_elevation[self.rx_id_onboard], (self.num_tx, 1)).T
        self.virtual_array_tx_id = np.tile(self.tx_id_transfer_order.reshape(1, -1), (self.num_rx, 1))
        self.virtual_array_rx_id = np.tile(self.rx_id_onboard.reshape(-1, 1), (1, self.num_tx))

        # azimuth, elevation, rx_id, tx_id
        self.virtual_array = np.hstack(
            (
                self.virtual_array_azimuth.reshape((-1, 1), order='F'),
                self.virtual_array_elevation.reshape((-1, 1), order='F'),
                self.virtual_array_rx_id.reshape((-1, 1), order='F'),
                self.virtual_array_tx_id.reshape((-1, 1), order='F')
            )
        )

        # get antenna_noredundant
        _, self.virtual_array_index_noredundant = np.unique(
            self.virtual_array[:, :2], axis=0, return_index=True
        )
        self.virtual_array_noredundant = self.virtual_array[self.virtual_array_index_noredundant, :]

        # get antenna_redundant
        self.virtual_array_index_redundant = np.setxor1d(
            np.arange(self.virtual_array.shape[0]),
            self.virtual_array_index_noredundant
        )
        self.virtual_array_redundant = self.virtual_array[self.virtual_array_index_redundant, :]

        # find and associate overlaped rx_tx pairs
        virtual_array_info_overlaped_associate = []
        for i in range(self.virtual_array_index_redundant.shape[0]):
            mask = (self.virtual_array_noredundant == self.virtual_array_redundant[i])
            mask = np.logical_and(mask[:, 0], mask[:, 1])
            info_associate = self.virtual_array_noredundant[mask][0]
            info_overlaped = self.virtual_array_redundant[i]
            virtual_array_info_overlaped_associate.append(
                np.concatenate((info_associate, info_overlaped)).tolist()
            )
        # azimuth, elevation, rx_associated, tx_associated, azimuth, elevation, rx_overlaped, tx_overlaped
        virtual_array_info_overlaped_associate = np.array(virtual_array_info_overlaped_associate)

        diff_tx = abs(virtual_array_info_overlaped_associate[:, 7] - virtual_array_info_overlaped_associate[:, 3])
        virtual_array_info_overlaped_diff1tx = virtual_array_info_overlaped_associate[diff_tx == 1]

        sorted_index = np.argsort(virtual_array_info_overlaped_diff1tx[:, 0])
        virtual_array_info_overlaped_diff1tx = virtual_array_info_overlaped_diff1tx[sorted_index, :]
        self.virtual_array_info_overlaped_diff1tx = virtual_array_info_overlaped_diff1tx

        # find noredundant row1
        self.virtual_array_noredundant_row1 = self.virtual_array_noredundant[self.virtual_array_noredundant[:, 1] == 0]

    def __rangeFFT__(self, window_enable=True, scale_on=False):
        self.data_rangeFFT = self.data_reordered

        # DC offset compensation
        self.data_rangeFFT = self.data_rangeFFT - self.data_rangeFFT.mean(axis=0)

        if window_enable:
            # use hanning window
            window_coeff_vec = np.hanning(self.rangeFFT_size+2)[1:-1]
            self.data_rangeFFT = self.data_rangeFFT * window_coeff_vec.reshape((-1, 1, 1, 1))

        self.data_rangeFFT = np.fft.fft(self.data_rangeFFT, n=self.rangeFFT_size, axis=0)

        if scale_on:
            scale_factor = self.scale_factor[int(np.log2(self.rangeFFT_size)-4)]
            self.data_rangeFFT *= scale_factor

    def __dopplerFFT__(self, window_enable=False, scale_on=False, clutter_remove=False):
        self.data_dopplerFFT = self.data_rangeFFT

        if window_enable:
            # use hanning window
            window_coeff_vec = np.hanning(self.dopplerFFT_size+2)[1:-1]
            self.data_dopplerFFT = self.data_dopplerFFT * window_coeff_vec.reshape((1, -1, 1, 1))

        if clutter_remove:
            self.data_dopplerFFT = self.data_dopplerFFT - np.expand_dims(self.data_dopplerFFT.mean(axis=1), axis=1)

        self.data_dopplerFFT = np.fft.fft(self.data_dopplerFFT, n=self.dopplerFFT_size, axis=1)
        self.data_dopplerFFT = np.fft.fftshift(self.data_dopplerFFT, axes=1)

        if scale_on:
            scale_factor = self.scale_factor[int(np.log2(self.dopplerFFT_size)-4)]
            self.data_rangeFFT *= scale_factor

    def __range_cfar_os__(self):
        refWinSize = 8
        guardWinSize = 8
        K0 = 5
        discardCellLeft = 10
        discardCellRight = 20
        maxEnable = 0
        sortSelectFactor = 0.75
        gaptot = refWinSize + guardWinSize
        n_obj = 0
        index_obj = []
        intensity_obj = []
        noise_obj = []
        snr_obj = []

        n_range = self.sig_integrate.shape[0]
        n_doppler = self.sig_integrate.shape[1]
        for i_doppler in range(n_doppler):
            sigv = self.sig_integrate[:, i_doppler]
            vecMid = sigv[discardCellLeft: n_range - discardCellRight]
            vecLeft = vecMid[0: gaptot]
            vecRight = vecMid[-gaptot:]
            vec = np.hstack((vecLeft, vecMid, vecRight))

            for j in range(len(vecMid)):
                index_cur = j + gaptot
                index_left = list(range(index_cur - gaptot, index_cur - guardWinSize))
                index_right = list(range(index_cur + guardWinSize + 1, index_cur + gaptot + 1))

                sorted_res = np.sort(np.hstack((vec[index_left], vec[index_right])), axis=0)
                value_selected = sorted_res[int(np.ceil(sortSelectFactor * len(sorted_res)) - 1)]

                if maxEnable == 1:
                    # whether value_selected is the local max value
                    value_local = vec[index_cur - gaptot: index_cur + gaptot + 1]
                    value_max = value_local.max()
                    if vec[index_cur] >= K0 * value_selected and vec[index_cur] >= value_max:
                        n_obj += 1
                        index_obj.append([discardCellLeft + j, i_doppler])
                        intensity_obj.append(vec[index_cur])
                        noise_obj.append(value_selected)
                        snr_obj.append(vec[index_cur] / value_selected)
                else:
                    if vec[index_cur] >= K0 * value_selected:
                        n_obj += 1
                        index_obj.append([discardCellLeft + j, i_doppler])
                        intensity_obj.append(vec[index_cur])
                        noise_obj.append(value_selected)
                        snr_obj.append(vec[index_cur] / value_selected)

        self.res_range_cfar = {
            'n_obj': n_obj,
            'index_obj': np.array(index_obj, dtype='int'),
            'intensity_obj': np.array(intensity_obj, dtype='float'),
            'noise_obj': np.array(noise_obj, dtype='float'),
            'snr_obj': np.array(snr_obj, dtype='float')
        }

    def __doppler_cfar_os_cyclicity__(self):
        refWinSize = 4
        guardWinSize = 0
        K0 = 0.5
        maxEnable = 0
        sortSelectFactor = 0.75
        gaptot = refWinSize + guardWinSize
        n_obj = 0
        index_obj = []
        intensity_obj = []
        noise_obj = []
        snr_obj = []

        index_obj_range = self.res_range_cfar['index_obj']
        index_obj_range_unique = np.unique(index_obj_range[:, 0])
        for i_range in index_obj_range_unique:
            sigv = self.sig_integrate[i_range, :]
            # cyclicity
            vecMid = sigv
            vecLeft = sigv[-gaptot:]
            vecRight = sigv[0: gaptot]
            vec = np.hstack((vecLeft, vecMid, vecRight))

            for j in range(len(vecMid)):
                index_cur = j + gaptot
                index_left = list(range(index_cur - gaptot, index_cur - guardWinSize))
                index_right = list(range(index_cur + guardWinSize + 1, index_cur + gaptot + 1))

                sorted_res = np.sort(np.hstack((vec[index_left], vec[index_right])), axis=0)
                value_selected = sorted_res[int(np.ceil(sortSelectFactor * len(sorted_res)) - 1)]

                if maxEnable == 1:
                    # whether value_selected is the local max value
                    value_local = vec[index_cur - gaptot: index_cur + gaptot + 1]
                    value_max = value_local.max()
                    if vec[index_cur] >= K0 * value_selected and vec[index_cur] >= value_max:
                        if j in index_obj_range[
                            index_obj_range[:, 0] == i_range, 1]:  # whether j also detected in range_cfar
                            n_obj += 1
                            index_obj.append([i_range, j])
                            intensity_obj.append(vec[index_cur])
                            noise_obj.append(value_selected)
                            snr_obj.append(vec[index_cur] / value_selected)
                else:
                    if vec[index_cur] >= K0 * value_selected:
                        if j in index_obj_range[index_obj_range[:, 0] == i_range, 1]:
                            n_obj += 1
                            index_obj.append([i_range, j])
                            intensity_obj.append(vec[index_cur])
                            noise_obj.append(value_selected)
                            snr_obj.append(vec[index_cur] / value_selected)

        self.res_doppler_cfar = {
            'n_obj': n_obj,
            'index_obj': np.array(index_obj, dtype='int'),
            'intensity_obj': np.array(intensity_obj, dtype='float'),
            'noise_obj': np.array(noise_obj, dtype='float'),
            'snr_obj': np.array(snr_obj, dtype='float')
        }

    def __rdm_cfar__(self):
        self.sig_integrate = np.sum(
            np.power(
                np.abs(
                    self.data_dopplerFFT.reshape(
                        (self.data_dopplerFFT.shape[0], self.data_dopplerFFT.shape[1], -1)
                    )
                ), 2
            ),
            axis=2
        ) + 1
        self.sig_integrate = np.asarray(self.sig_integrate)

        # do CFAR on range doppler map
        self.__range_cfar_os__()

        if self.res_range_cfar['n_obj'] > 0:
            self.__doppler_cfar_os_cyclicity__()

            n_obj = self.res_doppler_cfar['n_obj']

            range_index = self.res_doppler_cfar['index_obj'][:, 0]
            range_obj = range_index * self.infos['range_res_m']
            doppler_index = self.res_doppler_cfar['index_obj'][:, 1]
            doppler_obj = (doppler_index - self.data_dopplerFFT.shape[1] / 2) * self.infos['velocity_res_m_sec']

            # use the first noise estimate, apply to the second detection
            index1 = self.res_range_cfar['index_obj']
            index2 = self.res_doppler_cfar['index_obj']
            index_noise = [
                np.argwhere(np.logical_and(index1[:,0]==index2[i,0], index1[:,1]==index2[i,1]))[0][0]
                for i in range(index2.shape[0])
            ]
            noise = self.res_range_cfar['noise_obj'][index_noise]
            bin_val = self.data_dopplerFFT[range_index, doppler_index, :, :]
            intensity = np.power(np.abs(bin_val.reshape((bin_val.shape[0], -1))), 2).sum(axis=1)
            snr = 10 * np.log10(intensity / noise)

            self.res_rdm_cfar = {
                'n_obj': n_obj,
                'range_index': range_index,
                'range': range_obj,
                'doppler_index': doppler_index,
                'doppler': doppler_obj,
                'doppler_index_origin': doppler_index,
                'doppler_origin': doppler_obj,
                'noise': noise,
                'bin_val': bin_val,
                'intensity': intensity,
                'snr': snr
            }

            # doppler phase correction due to TDM MIMO with applying vmax extention algorithm
            if self.apply_vmax_extend:
                doppler_index_unwrap = doppler_index.reshape((-1, 1)) + self.dopplerFFT_size * \
                                       (np.arange(self.num_tx).reshape((1, -1)) - self.num_tx / 2 + 1 * (doppler_obj <= 0).reshape((-1, 1)))
                doppler_index_unwrap = doppler_index_unwrap.astype(np.int)

                sig_bin_org = bin_val

                # Doppler phase correction due to TDM MIMO
                delta_phi = 2 * np.pi * (doppler_index_unwrap - self.dopplerFFT_size / 2) / (self.num_tx * self.dopplerFFT_size)

                # construct all possible signal vectors based on the number of possible hypothesis
                correct_matrix = np.exp(
                    -1j * np.arange(self.num_tx).reshape((1, 1, -1, 1)) *
                    np.expand_dims(np.expand_dims(delta_phi, axis=1), axis=1)
                )
                sig_bin = np.expand_dims(sig_bin_org, axis=-1) * correct_matrix

                # use overlap antenna to do max doppler unwrap
                index_antenna_overlaped_diff1tx = np.stack(
                    (
                        np.array([np.argwhere(self.rx_id_onboard == i)[0][0] for i in self.virtual_array_info_overlaped_diff1tx[:, 2]], dtype='int'),
                        np.array([np.argwhere(self.tx_id_transfer_order == i)[0][0] for i in self.virtual_array_info_overlaped_diff1tx[:, 3]], dtype='int'),
                        np.array([np.argwhere(self.rx_id_onboard == i)[0][0] for i in self.virtual_array_info_overlaped_diff1tx[:, 6]], dtype='int'),
                        np.array([np.argwhere(self.tx_id_transfer_order==i)[0][0] for i in self.virtual_array_info_overlaped_diff1tx[:, 7]], dtype='int'),
                    ),
                    axis=1
                )
                sig_overlap = np.stack(
                    (
                        sig_bin_org[:, index_antenna_overlaped_diff1tx[:, 0], index_antenna_overlaped_diff1tx[:, 1]],
                        sig_bin_org[:, index_antenna_overlaped_diff1tx[:, 2], index_antenna_overlaped_diff1tx[:, 3]]
                    ),
                    axis=2
                )

                # check the phase difference of each overlap antenna pair for each hypothesis
                angle_sum_test = np.zeros((sig_overlap.shape[0], sig_overlap.shape[1], delta_phi.shape[1]))
                for i_sig in range(angle_sum_test.shape[1]):
                    signal2 = np.matmul(
                        np.expand_dims(sig_overlap[:, :i_sig+1, 1], axis=2),
                        np.expand_dims(np.exp(-1j * delta_phi), axis=1)
                    )
                    angle_sum_test[:, i_sig, :] = np.angle(
                        np.sum(
                            np.expand_dims(sig_overlap[:, :i_sig+1, 0], axis=2) * signal2.conj(),
                            axis=1
                        )
                    )

                # chosee the hypothesis with minimum phase difference to estimate the unwrap factor
                doppler_unwrap_integ_overlap_index = np.argmin(np.abs(angle_sum_test), axis=2)

                # test the angle FFT SNR
                index_antenna_noredundant_row1 = np.stack(
                    (
                        np.array([np.argwhere(self.rx_id_onboard == i)[0][0] for i in self.virtual_array_noredundant_row1[:, 2]], dtype='int'),
                        np.array([np.argwhere(self.tx_id_transfer_order == i)[0][0] for i in self.virtual_array_noredundant_row1[:, 3]], dtype='int')
                    ),
                    axis=1
                )
                sig_bin_row1 = sig_bin[:, index_antenna_noredundant_row1[:, 0], index_antenna_noredundant_row1[:, 1], :]
                angleFFT_size = 128
                sig_bin_row1_fft = np.fft.fftshift(
                    np.fft.fft(sig_bin_row1, n=angleFFT_size, axis=1),
                    axes=1
                )
                angle_bin_skip_left = 4
                angle_bin_skip_right = 4
                sig_bin_row1_fft_cut = np.abs(
                    sig_bin_row1_fft[:, angle_bin_skip_left:angleFFT_size-angle_bin_skip_right, :]
                )
                doppler_unwrap_integ_FFT_index = np.argmax(
                    np.max(sig_bin_row1_fft_cut, axis=1),
                    axis=1
                )

                doppler_unwrap_integ_index = np.array([
                    np.argmax(np.bincount(doppler_unwrap_integ_overlap_index[i, :]))
                    for i in range(doppler_unwrap_integ_overlap_index.shape[0])
                ])
                self.res_rdm_cfar['bin_val_correct'] = sig_bin[np.arange(sig_bin.shape[0]), :, :, doppler_unwrap_integ_index]

                # corret doppler after applying the integer value
                doppler_index_correct = doppler_index_unwrap[np.arange(doppler_index_unwrap.shape[0]), doppler_unwrap_integ_index]
                doppler_index_correct_FFT = doppler_index_unwrap[np.arange(doppler_index_unwrap.shape[0]), doppler_unwrap_integ_FFT_index]
                doppler_index_correct_overlap = doppler_index_unwrap[np.arange(doppler_index_unwrap.shape[0]), doppler_unwrap_integ_index]
                self.res_rdm_cfar['doppler_index_correct'] = doppler_index_correct
                self.res_rdm_cfar['doppler_correct'] = (doppler_index_correct - self.data_dopplerFFT.shape[1] / 2) * \
                                                       self.infos['velocity_res_m_sec']
                self.res_rdm_cfar['doppler_index_correct_FFT'] = doppler_index_correct_FFT
                self.res_rdm_cfar['doppler_correct_FFT'] = (doppler_index_correct_FFT - self.data_dopplerFFT.shape[1] / 2) * \
                                                           self.infos['velocity_res_m_sec']
                self.res_rdm_cfar['doppler_index_correct_overlap'] = doppler_index_correct_overlap
                self.res_rdm_cfar['doppler_correct_overlap'] = (doppler_index_correct_overlap - self.data_dopplerFFT.shape[1] / 2) * \
                                                               self.infos['velocity_res_m_sec']
                # replace doppler_index: doppler_index_correct, doppler: doppler_correct
                self.res_rdm_cfar['doppler_index'] = self.res_rdm_cfar['doppler_index_correct']
                self.res_rdm_cfar['doppler'] = self.res_rdm_cfar['doppler_correct']

                # apply min_dis_apply_vmax_extend
                index_noapply = np.argwhere(range_obj <= self.min_dis_apply_vmax_extend)[:, 0]

                # doppler phase correction due to TDM MIMO without applying vmax extention algorithm
                delta_phi_noapply = (
                        2 * np.pi * (doppler_index - self.dopplerFFT_size / 2) / (self.num_tx * self.dopplerFFT_size)
                ).reshape((-1, 1))
                i_tx_noapply = np.arange(self.num_tx).reshape((1, -1))
                sig_bin_noapply = bin_val * np.exp(
                    -1j * np.matmul(delta_phi_noapply, i_tx_noapply)
                ).reshape((delta_phi_noapply.shape[0], 1, i_tx_noapply.shape[1]))

                self.res_rdm_cfar['bin_val_correct'][index_noapply, :, :] = sig_bin_noapply[index_noapply, :, :]
                self.res_rdm_cfar['doppler_index_correct'][index_noapply] = self.res_rdm_cfar['doppler_index_origin'][index_noapply]
                self.res_rdm_cfar['doppler_correct'][index_noapply] = self.res_rdm_cfar['doppler_origin'][index_noapply]
                self.res_rdm_cfar['doppler_index_correct_FFT'][index_noapply] = self.res_rdm_cfar['doppler_index_origin'][index_noapply]
                self.res_rdm_cfar['doppler_correct_FFT'][index_noapply] = self.res_rdm_cfar['doppler_origin'][index_noapply]
                self.res_rdm_cfar['doppler_index_correct_overlap'][index_noapply] = self.res_rdm_cfar['doppler_index_origin'][index_noapply]
                self.res_rdm_cfar['doppler_correct_overlap'][index_noapply] = self.res_rdm_cfar['doppler_origin'][index_noapply]
                self.res_rdm_cfar['doppler_index'][index_noapply] = self.res_rdm_cfar['doppler_index_origin'][index_noapply]
                self.res_rdm_cfar['doppler'][index_noapply] = self.res_rdm_cfar['doppler_origin'][index_noapply]

            else:
                # doppler phase correction due to TDM MIMO without applying vmax extention algorithm
                sig_bin_org = bin_val

                delta_phi = (
                        2 * np.pi * (doppler_index - self.data_dopplerFFT.shape[1] / 2) / \
                        (self.data_dopplerFFT.shape[3] * self.data_dopplerFFT.shape[1])
                ).reshape((-1, 1))

                i_tx = np.arange(self.data_dopplerFFT.shape[3]).reshape((1, -1))

                correct_matrix = np.exp(-1j * np.matmul(delta_phi, i_tx)).reshape(
                    (delta_phi.shape[0], 1, i_tx.shape[1])
                )

                sig_bin_correct = sig_bin_org * correct_matrix

                self.res_rdm_cfar['bin_val_correct'] = sig_bin_correct
        else:
            self.res_rdm_cfar = None

    def __single_obj_signal_space_mapping__(self, sig):
        sig_space = np.zeros(
            (self.virtual_array_azimuth.max() + 1, self.virtual_array_elevation.max() + 1),
            dtype=sig.dtype
        )
        sig_space_index0 = self.virtual_array_noredundant[:, 0]
        sig_space_index1 = self.virtual_array_noredundant[:, 1]
        sig_index0 = np.array([np.argwhere(self.rx_id_onboard == i_rx)[0][0] for i_rx in self.virtual_array_noredundant[:, 2]])
        sig_index1 = np.array([np.argwhere(self.tx_id_transfer_order == i_tx)[0][0] for i_tx in self.virtual_array_noredundant[:, 3]])
        sig_space[sig_space_index0, sig_space_index1] = sig[sig_index0, sig_index1]
        return sig_space

    def __DOA_BF_PeakDet_loc__(self, input_data, sidelobeLevel_dB):
        gamma = np.power(10, 0.02)

        min_val = np.inf
        max_val = -np.inf
        max_loc = 0
        max_data = []
        locate_max = 0
        num_max = 0
        extend_loc = 0
        init_stage = 1
        abs_max_value = 0

        i = -1
        N = len(input_data)
        while (i < (N + extend_loc - 1)):
            i += 1
            i_loc = np.mod(i, N)
            current_val = input_data[i_loc]
            # record the maximum abs value
            if current_val > abs_max_value:
                abs_max_value = current_val
            # record the maximum value and loc
            if current_val > max_val:
                max_val = current_val
                max_loc = i_loc
                max_loc_r = i
            # record the minimum value
            if current_val < min_val:
                min_val = current_val
            if locate_max == 1:
                if current_val < max_val / gamma:
                    num_max += 1
                    bwidth = i - max_loc_r
                    max_data.append([max_loc, max_val, bwidth, max_loc_r])
                    min_val = current_val
                    locate_max = 0
            else:
                if current_val > min_val * gamma:
                    locate_max = 1
                    max_val = current_val
                    max_loc = i_loc
                    max_loc_r = i

                    if init_stage == 1:
                        extend_loc = i
                        init_stage = 0


        max_data = np.array(max_data)
        if len(max_data.shape) < 2:
            max_data = np.zeros((0, 4))

        max_data = max_data[max_data[:, 1] >= abs_max_value * pow(10, -sidelobeLevel_dB / 10), :]

        peak_val = max_data[:, 1]
        peak_loc = max_data[:, 0].astype('int')
        return peak_val, peak_loc

    def __DOA_beamformingFFT_2D__(self, sig):
        sidelobeLevel_dB_azim = 1
        sidelobeLevel_dB_elev = 0
        doa_fov_azim = [-70, 70]
        doa_fov_elev = [-20, 20]

        data_space = self.__single_obj_signal_space_mapping__(sig)
        data_azimuthFFT = np.fft.fftshift(np.fft.fft(data_space, n=self.azimuthFFT_size, axis=0), axes=0)
        data_elevationFFT = np.fft.fftshift(np.fft.fft(data_azimuthFFT, n=self.elevationFFT_size, axis=1), axes=1)

        spec_azim = np.abs(data_azimuthFFT[:, 0])
        _, peak_loc_azim = self.__DOA_BF_PeakDet_loc__(spec_azim, sidelobeLevel_dB_azim)

        n_obj = 0
        azimuth_obj = []
        elevation_obj = []
        azimuth_index = []
        elevation_index = []
        for i in range(len(peak_loc_azim)):
            spec_elev = abs(data_elevationFFT[peak_loc_azim[i], :])
            peak_val_elev, peak_loc_elev = self.__DOA_BF_PeakDet_loc__(spec_elev, sidelobeLevel_dB_elev)
            for j in range(len(peak_loc_elev)):
                est_azimuth = np.arcsin(self.azimuth_bins[peak_loc_azim[i]] / 2 / np.pi / self.doa_unitDis) / np.pi * 180
                est_elevation = np.arcsin(self.elevation_bins[peak_loc_elev[j]] / 2 / np.pi / self.doa_unitDis) / np.pi * 180
                if est_azimuth >= doa_fov_azim[0] and est_azimuth <= doa_fov_azim[1] and est_elevation >= doa_fov_elev[0] and est_elevation <= doa_fov_elev[1]:
                    n_obj += 1
                    azimuth_obj.append(est_azimuth)
                    elevation_obj.append(est_elevation)
                    azimuth_index.append(peak_loc_azim[i])
                    elevation_index.append(peak_loc_elev[j])

        azimuth_obj = np.array(azimuth_obj, dtype='float')
        elevation_obj = np.array(elevation_obj, dtype='float')
        azimuth_index = np.array(azimuth_index, dtype='int')
        elevation_index = np.array(elevation_index, dtype='int')

        res = {
            'data_space': data_space,
            'data_azimuthFFT': data_azimuthFFT,
            'data_elevationFFT': data_elevationFFT,
            'n_obj': n_obj,
            'azimuth_obj': azimuth_obj,
            'elevation_obj': elevation_obj,
            'azimuth_index': azimuth_index,
            'elevation_index': elevation_index
        }

        return res

    def __doa__(self):
        n_obj = 0
        range_index = []
        range_val = []
        doppler_index = []
        doppler = []
        if self.apply_vmax_extend:
            doppler_index_origin = []
            doppler_origin = []
        azimuth_index = []
        azimuth = []
        elevation_index = []
        elevation = []
        snr = []
        intensity = []
        noise = []

        for i in range(self.res_rdm_cfar['n_obj']):
            x = self.res_rdm_cfar['bin_val_correct'][i, :, :]
            res_DOA_beamformingFFT_2D = self.__DOA_beamformingFFT_2D__(x)

            for j in range(res_DOA_beamformingFFT_2D['n_obj']):
                n_obj += 1
                range_index.append(self.res_rdm_cfar['range_index'][i])
                range_val.append(self.res_rdm_cfar['range'][i])
                doppler_index.append(self.res_rdm_cfar['doppler_index'][i])
                doppler.append(self.res_rdm_cfar['doppler'][i])
                if self.apply_vmax_extend:
                    doppler_index_origin.append(self.res_rdm_cfar['doppler_index_origin'][i])
                    doppler_origin.append(self.res_rdm_cfar['doppler_origin'][i])

                azimuth_index.append(res_DOA_beamformingFFT_2D['azimuth_index'][j])
                azimuth.append(res_DOA_beamformingFFT_2D['azimuth_obj'][j])
                elevation_index.append(res_DOA_beamformingFFT_2D['elevation_index'][j])
                elevation.append(res_DOA_beamformingFFT_2D['elevation_obj'][j])

                snr.append(self.res_rdm_cfar['snr'][i])
                intensity.append(self.res_rdm_cfar['intensity'][i])
                noise.append(self.res_rdm_cfar['noise'][i])

        if n_obj > 0:
            self.res_doa = {
                'n_obj': n_obj,
                'range_index': np.array(range_index, dtype='int'),
                'range': np.array(range_val, dtype='float'),
                'doppler_index': np.array(doppler_index, dtype='int'),
                'doppler': np.array(doppler, dtype='float'),
                'azimuth_index': np.array(azimuth_index, dtype='int'),
                'azimuth': np.array(azimuth, dtype='float'),
                'elevation_index': np.array(elevation_index, dtype='int'),
                'elevation': np.array(elevation, dtype='float'),
                'snr': np.array(snr, dtype='float'),
                'intensity': np.array(intensity, dtype='float'),
                'noise': np.array(noise, dtype='float')
            }
            if self.apply_vmax_extend:
                self.res_doa['doppler_index_origin'] = np.array(doppler_index_origin, dtype='int')
                self.res_doa['doppler_origin'] = np.array(doppler_origin, dtype='float')
        else:
            self.res_doa = None

    def __generate_pcd__(self):
        r = self.res_doa['range']
        azi = self.res_doa['azimuth']
        ele = self.res_doa['elevation']

        x = r * np.cos(ele / 180 * np.pi) * np.sin(azi / 180 * np.pi)
        y = r * np.cos(ele / 180 * np.pi) * np.cos(azi / 180 * np.pi)
        z = r * np.sin(ele / 180 * np.pi)
        doppler = self.res_doa['doppler']
        snr = self.res_doa['snr']
        intensity = self.res_doa['intensity']
        noise = self.res_doa['noise']

        self.pcd = {
            'x': x,
            'y': y,
            'z': z,
            'doppler': doppler,
            'snr': snr,
            'intensity': intensity,
            'noise': noise
        }

    def __generate_heatmapBEV__(self, doppler_correction=False):
        sig = self.data_dopplerFFT

        if doppler_correction:
            # add Doppler correction before generating the heatmap
            delta_phi = 2 * np.pi * (np.arange(self.dopplerFFT_size) - self.dopplerFFT_size/2) /\
                        (self.num_tx * self.dopplerFFT_size)
            cor_vec = np.exp(-1j * delta_phi.reshape((-1, 1)) * np.arange(self.num_tx).reshape((1, -1)))

            sig = sig * np.expand_dims(np.expand_dims(cor_vec, axis=1), axis=0)

        index_antenna_noredundant_row1 = np.stack(
            (
                np.array([np.argwhere(self.rx_id_onboard == i)[0][0] for i in self.virtual_array_noredundant_row1[:, 2]], dtype='int'),
                np.array([np.argwhere(self.tx_id_transfer_order == i)[0][0] for i in self.virtual_array_noredundant_row1[:, 3]], dtype='int')
            ),
            axis=1
        )
        sig_row1 = sig[:, :, index_antenna_noredundant_row1[:, 0], index_antenna_noredundant_row1[:, 1]]
        sig_row1_azimuthFFT = np.fft.fftshift(
            np.fft.fft(sig_row1, n=self.azimuthFFT_size, axis=2),
            axes=2
        )

        heatmapBEV_static = np.abs(sig_row1_azimuthFFT[:, self.dopplerFFT_size // 2, :])
        heatmapBEV_static = np.hstack((heatmapBEV_static, heatmapBEV_static[:, 0:1]))
        heatmapBEV_dynamic = np.sum(
            np.abs(sig_row1_azimuthFFT[:, np.arange(self.dopplerFFT_size) != self.dopplerFFT_size // 2, :]),
            axis=1
        )
        heatmapBEV_dynamic = np.hstack((heatmapBEV_dynamic, heatmapBEV_dynamic[:, 0:1]))

        sine_theta = np.arange(1, -1-2/self.azimuthFFT_size, -2/self.azimuthFFT_size)
        cos_theta = np.sqrt(1 - np.power(sine_theta, 2))
        x = np.matmul(
            self.range_bins.reshape((-1, 1)),
            sine_theta.reshape((1, -1))
        )
        y = np.matmul(
            self.range_bins.reshape((-1, 1)),
            cos_theta.reshape((1, -1))
        )

        self.heatmapBEV = {
            'heatmapBEV_static': heatmapBEV_static,
            'heatmapBEV_dynamic': heatmapBEV_dynamic,
            'x': x,
            'y': y
        }

    def __generate_heatmap4D__(self, doppler_correction=False):
        pass
        # sig = self.data_dopplerFFT
        #
        # if doppler_correction:
        #     # add Doppler correction before generating the heatmap
        #     delta_phi = 2 * np.pi * (np.arange(self.dopplerFFT_size) - self.dopplerFFT_size / 2) / \
        #                 (self.num_tx * self.dopplerFFT_size)
        #     cor_vec = np.exp(-1j * delta_phi.reshape((-1, 1)) * np.arange(self.num_tx).reshape((1, -1)))
        #
        #     sig = sig * np.expand_dims(np.expand_dims(cor_vec, axis=1), axis=0)
        #
        # sig_space = np.zeros(
        #     (self.rangeFFT_size, self.dopplerFFT_size, self.virtual_array_azimuth.max() + 1, self.virtual_array_elevation.max() + 1),
        #     dtype=sig.dtype
        # )
        # sig_space_index0 = self.virtual_array_noredundant[:, 0]
        # sig_space_index1 = self.virtual_array_noredundant[:, 1]
        # sig_index0 = np.array(
        #     [np.argwhere(self.rx_id_onboard == i_rx)[0][0] for i_rx in self.virtual_array_noredundant[:, 2]])
        # sig_index1 = np.array(
        #     [np.argwhere(self.tx_id_transfer_order == i_tx)[0][0] for i_tx in self.virtual_array_noredundant[:, 3]])
        # sig_space[:, :, sig_space_index0, sig_space_index1] = sig[:, :, sig_index0, sig_index1]
        #
        # sig_space = cp.asarray(sig_space)
        # sig_space_azimuthFFT = cp.fft.fftshift(cp.fft.fft(sig_space, n=self.azimuthFFT_size, axis=2), axes=2)
        # sig_space_elevationFFT = cp.fft.fftshift(cp.fft.fft(sig_space_azimuthFFT, n=self.elevationFFT_size, axis=3), axes=3)
        # heatmap4D = cp.abs(sig_space_elevationFFT)
        # heatmap4D = cp.asnumpy(heatmap4D)
        #
        # r = self.range_bins
        # azi = np.arcsin(self.azimuth_bins / 2 / np.pi / self.doa_unitDis) / np.pi * 180
        # ele = np.arcsin(self.elevation_bins / 2 / np.pi / self.doa_unitDis) / np.pi * 180
        # x = r * np.cos(ele / 180 * np.pi) * np.sin(azi / 180 * np.pi)
        # y = r * np.cos(ele / 180 * np.pi) * np.cos(azi / 180 * np.pi)
        # z = r * np.sin(ele / 180 * np.pi)
        # doppler = self.doppler_bins
        #
        # self.heatmap4D = {
        #     'heatmap4D': heatmap4D,
        #     'x': x,
        #     'y': y,
        #     'z': z,
        #     'doppler': doppler
        # }
