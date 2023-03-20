function obj = doa(obj)
    n_obj = 0;
    range_index = [];
    range_val = [];
    doppler_index = [];
    doppler = [];
    if obj.apply_vmax_extend
        doppler_index_origin = [];
        doppler_origin = [];
    end
    azimuth_index = [];
    azimuth = [];
    elevation_index = [];
    elevation = [];
    snr = [];
    intensity = [];
    noise = [];

    for i = 1:obj.res_rdm_cfar.n_obj
        x = squeeze(obj.res_rdm_cfar.bin_val_correct(i, :, :));
        res_DOA_beamformingFFT_2D = DOA_beamformingFFT_2D(obj, x);

        for j = 1:res_DOA_beamformingFFT_2D.n_obj
            n_obj = n_obj + 1;
            range_index = [range_index; obj.res_rdm_cfar.range_index(i)];
            range_val = [range_val; obj.res_rdm_cfar.range(i)];
            doppler_index = [doppler_index; obj.res_rdm_cfar.doppler_index(i)];
            doppler = [doppler; obj.res_rdm_cfar.doppler(i)];
            if obj.apply_vmax_extend
                doppler_index_origin = [doppler_index_origin; obj.res_rdm_cfar.doppler_index_origin(i)];
                doppler_origin = [doppler_origin; obj.res_rdm_cfar.doppler_origin(i)];
            end
            azimuth_index = [azimuth_index; res_DOA_beamformingFFT_2D.azimuth_index(j)];
            azimuth = [azimuth; res_DOA_beamformingFFT_2D.azimuth_obj(j)];
            elevation_index = [elevation_index; res_DOA_beamformingFFT_2D.elevation_index(j)];
            elevation = [elevation; res_DOA_beamformingFFT_2D.elevation_obj(j)];

            snr = [snr; obj.res_rdm_cfar.snr(i)];
            intensity = [intensity; obj.res_rdm_cfar.intensity(i)];
            noise = [noise; obj.res_rdm_cfar.noise(i)];
        end
    end
    
    
    if n_obj > 0
        obj.res_doa = struct();
        obj.res_doa.n_obj = n_obj;
        obj.res_doa.range_index = range_index;
        obj.res_doa.range = range_val;
        obj.res_doa.doppler_index = doppler_index;
        obj.res_doa.doppler = doppler;
        if obj.apply_vmax_extend
            obj.res_doa.doppler_index_origin = doppler_index_origin;
            obj.res_doa.doppler_origin = doppler_origin;
        end
        obj.res_doa.azimuth_index = azimuth_index;
        obj.res_doa.azimuth = azimuth;
        obj.res_doa.elevation_index = elevation_index;
        obj.res_doa.elevation = elevation;
        obj.res_doa.snr = snr;
        obj.res_doa.intensity = intensity;
        obj.res_doa.noise = noise;
    else
        obj.res_doa = struct();
    end
end

function res_DOA_beamformingFFT_2D = DOA_beamformingFFT_2D(obj, sig)
    sidelobeLevel_dB_azim = 1;
    sidelobeLevel_dB_elev = 0;
    doa_fov_azim = [-70, 70];
    doa_fov_elev = [-20, 20];

    data_space = single_obj_signal_space_mapping(obj, sig);
    data_azimuthFFT = fftshift(fft(data_space, obj.azimuthFFT_size, 1), 1);
    data_elevationFFT = fftshift(fft(data_azimuthFFT, obj.elevationFFT_size, 2), 2);

    spec_azim = abs(data_azimuthFFT(:, 1));
    [~, peak_loc_azim] = DOA_BF_PeakDet_loc(spec_azim, sidelobeLevel_dB_azim);

    n_obj = 0;
    azimuth_obj = [];
    elevation_obj = [];
    azimuth_index = [];
    elevation_index = [];
    for i = 1:size(peak_loc_azim, 1)
        spec_elev = abs(data_elevationFFT(peak_loc_azim(i), :));
        [~, peak_loc_elev] = DOA_BF_PeakDet_loc(spec_elev.', sidelobeLevel_dB_elev);
        for j = 1:size(peak_loc_elev, 1)
            est_azimuth = asin(obj.azimuth_bins(peak_loc_azim(i)) / 2 / pi / obj.doa_unitDis) / pi * 180;
            est_elevation = asin(obj.elevation_bins(peak_loc_elev(j)) / 2 / pi / obj.doa_unitDis) / pi * 180;
            if est_azimuth >= doa_fov_azim(1) && est_azimuth <= doa_fov_azim(2) ...
                && est_elevation >= doa_fov_elev(1) && est_elevation <= doa_fov_elev(2)
                n_obj = n_obj + 1;
                azimuth_obj = [azimuth_obj; est_azimuth];
                elevation_obj = [elevation_obj; est_elevation];
                azimuth_index = [azimuth_index; peak_loc_azim(i)];
                elevation_index = [elevation_index; peak_loc_elev(j)];
            end
        end
    end

    res_DOA_beamformingFFT_2D = struct();
    res_DOA_beamformingFFT_2D.data_space = data_space;
    res_DOA_beamformingFFT_2D.data_azimuthFFT = data_azimuthFFT;
    res_DOA_beamformingFFT_2D.data_elevationFFT = data_elevationFFT;
    res_DOA_beamformingFFT_2D.n_obj = n_obj;
    res_DOA_beamformingFFT_2D.azimuth_obj = azimuth_obj;
    res_DOA_beamformingFFT_2D.elevation_obj = elevation_obj;
    res_DOA_beamformingFFT_2D.azimuth_index = azimuth_index;
    res_DOA_beamformingFFT_2D.elevation_index = elevation_index;
end

function sig_space = single_obj_signal_space_mapping(obj, sig)
    n_azimuth = max(obj.virtual_array_azimuth, [], "all") + 1;
    n_elevation = max(obj.virtual_array_elevation, [], "all") + 1;
    sig_space = zeros(n_azimuth, n_elevation);
    
    virtual_array_noredundant = obj.virtual_array_noredundant;
    rx_id_onboard=  obj.rx_id_onboard;
    tx_id_transfer_order = obj.tx_id_transfer_order;
    for i = 1:size(virtual_array_noredundant, 1)
        idx_azimuth = virtual_array_noredundant(i, 1) + 1;
        idx_elevation = virtual_array_noredundant(i, 2) + 1;
        idx_rx = find(rx_id_onboard == virtual_array_noredundant(i, 3));
        idx_tx = find(tx_id_transfer_order == virtual_array_noredundant(i, 4));
        sig_space(idx_azimuth, idx_elevation) = sig(idx_rx, idx_tx);
    end
end

function [peak_val, peak_loc] = DOA_BF_PeakDet_loc(input_data, sidelobeLevel_dB)
    gamma = power(10, 0.02);
    min_val = inf;
    max_val = -inf;
    max_loc = 0;
    max_data = [];
    locate_max = 0;
    num_max = 0;
    extend_loc = 0;
    init_stage = 1;
    abs_max_value = 0;

    i = 0;
    N = size(input_data, 1);
    while(i < (N + extend_loc - 1))
        i = i + 1;
        i_loc = mod(i, N) + 1;
        current_val = input_data(i_loc);
        % record the maximum abs value
        if current_val > abs_max_value
            abs_max_value = current_val;
        end
        % record the maximum value and loc
        if current_val > max_val
            max_val = current_val;
            max_loc = i_loc;
            max_loc_r = i;
        end
        % record the minimum value
        if current_val < min_val
            min_val = current_val;
        end
        if locate_max == 1
            if current_val < max_val / gamma
                num_max = num_max + 1;
                bwidth = i - max_loc_r;
                max_data = [max_data; [max_loc, max_val, bwidth, max_loc_r]];
                min_val = current_val;
                locate_max = 0;
            end
        else
            if current_val > min_val * gamma
                locate_max = 1;
                max_val = current_val;
                max_loc = i_loc;
                max_loc_r = i;

                if init_stage == 1
                    extend_loc = i;
                    init_stage = 0;
                end
            end
        end
    end

    if ~isempty(max_data)
        max_data = max_data(max_data(:, 2) >= abs_max_value * power(10, -sidelobeLevel_dB / 10), :);
        peak_val = max_data(:, 2);
        peak_loc = max_data(:, 1);
    else
        peak_val = [];
        peak_loc = [];
    end
end