function obj = dopplerFFT(obj)
    data_dopplerFFT = obj.data_rangeFFT;

    if obj.dopplerFFT_window_enable
        % use hanning window
        window_coeff_vec = np.hanning(obj.dopplerFFT_size)';
        data_dopplerFFT = data_dopplerFFT .* window_coeff_vec;
    end

    if obj.dopplerFFT_clutter_remove
        data_dopplerFFT = data_dopplerFFT - mean(data_dopplerFFT, 2);
    end

    data_dopplerFFT = fft(data_dopplerFFT, obj.dopplerFFT_size, 2);
    data_dopplerFFT = fftshift(data_dopplerFFT, 2);

    if obj.dopplerFFT_scale_on
        scale_factor = obj.scale_factor(log2(dopplerFFT_size) - 3);
        data_dopplerFFT = scale_factor .* data_dopplerFFT;
    end

    obj.data_dopplerFFT = data_dopplerFFT;
end