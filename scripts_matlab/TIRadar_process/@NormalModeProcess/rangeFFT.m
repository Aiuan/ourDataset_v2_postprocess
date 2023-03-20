function obj = rangeFFT(obj)    
    data_rangeFFT = obj.data_reordered;

    % DC offset compensation
    data_rangeFFT = data_rangeFFT - mean(data_rangeFFT, 1);

    if obj.rangeFFT_window_enable
        % use hanning window       
        window_coeff_vec = hanning(obj.rangeFFT_size);
        data_rangeFFT = data_rangeFFT .* window_coeff_vec;
    end

    data_rangeFFT = fft(data_rangeFFT, obj.rangeFFT_size, 1);

    if obj.rangeFFT_scale_on
        scale_factor = obj.scale_factor(log2(obj.rangeFFT_size) - 3);
        data_rangeFFT = scale_factor .* data_rangeFFT;
    end

    obj.data_rangeFFT = data_rangeFFT;
end