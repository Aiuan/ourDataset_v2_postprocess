function obj = calibrate(obj)
    adc_sample_rate = obj.mode_infos.digOutSampleRate_ksps * 1e3;
    chirp_slope = obj.mode_infos.freqSlopeConst_MHz_usec * 1e6 / 1e-6;
    num_sample = double(obj.mode_infos.numAdcSamples);
    num_loop = double(obj.mode_infos.numLoops);
    num_rx = double(obj.num_rx);
    num_tx = double(obj.num_tx);

    range_mat = obj.calibmat.calibResult.RangeMat;
    fs_calib = obj.calibmat.params.Sampling_Rate_sps;
    slope_calib = obj.calibmat.params.Slope_MHzperus * 1e6 / 1e-6;
    calibration_interp = 5;
    peak_val_mat = obj.calibmat.calibResult.PeakValMat;
    phase_calib_only = 1;

    tx_id_ref = obj.tx_id_transfer_order(1);

    % construct the frequency compensation matrix
    freq_calib = 2 * pi * ( ...
        (range_mat(obj.tx_id_transfer_order, :) - range_mat(tx_id_ref, 1))...
        * fs_calib / adc_sample_rate * chirp_slope / slope_calib ...
        ) / (num_sample * calibration_interp);

    freq_correction_mat = conj(exp( ...
        1j * repmat((0:num_sample-1).', 1, num_loop, num_rx, num_tx) .* reshape(freq_calib.', 1, 1, num_rx, num_tx) ...
        ));

    phase_calib = peak_val_mat(tx_id_ref, 1) ./ peak_val_mat(obj.tx_id_transfer_order, :);
    if phase_calib_only == 1
        phase_calib = phase_calib ./ abs(phase_calib);
    end
    phase_correction_mat = reshape(phase_calib.', 1, 1, num_rx, num_tx);

    obj.data_calib = obj.data_raw .* freq_correction_mat .* phase_correction_mat;

end