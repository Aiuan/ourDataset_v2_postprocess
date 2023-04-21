function res = generate_heatmap_bev(obj)
    sig_rangeFFT = obj.data_rangeFFT;    
    sig_space = signal_space_mapping(obj, sig_rangeFFT);    
    sig_bev = sig_space(:, :, :, 1);
    
    azimuthFFT_size = obj.azimuthFFT_size;
    heatmap_bev = fftshift(fft(sig_bev, azimuthFFT_size, 3), 3);

    heatmap_bev = squeeze(sum(abs(heatmap_bev), 2));
%     heatmap_bev = squeeze(abs(heatmap_bev(:, 1, :)));

    heatmap_bev(:, end+1) = heatmap_bev(:, 1);
    
    sine_theta = 1: -2/azimuthFFT_size:-1;
    cos_theta = sqrt(1 - sine_theta.^2);
    range_bin = obj.range_bins;

    x = range_bin.' * sine_theta;
    y = range_bin.' * cos_theta;

    res = struct('value', heatmap_bev, 'x', x, 'y', y);
end

function sig_space = signal_space_mapping(obj, sig)
    n1 = size(sig, 1);
    n2 = size(sig, 2);
    n_azimuth = max(obj.virtual_array_azimuth, [], "all") + 1;
    n_elevation = max(obj.virtual_array_elevation, [], "all") + 1;
    sig_space = zeros(n1, n2, n_azimuth, n_elevation);
    
    virtual_array_noredundant = obj.virtual_array_noredundant;
    rx_id_onboard=  obj.rx_id_onboard;
    tx_id_transfer_order = obj.tx_id_transfer_order;
    for i = 1:size(virtual_array_noredundant, 1)
        idx_azimuth = virtual_array_noredundant(i, 1) + 1;
        idx_elevation = virtual_array_noredundant(i, 2) + 1;
        idx_rx = find(rx_id_onboard == virtual_array_noredundant(i, 3));
        idx_tx = find(tx_id_transfer_order == virtual_array_noredundant(i, 4));
        sig_space(:, :, idx_azimuth, idx_elevation) = sig(:, :, idx_rx, idx_tx);
    end
end