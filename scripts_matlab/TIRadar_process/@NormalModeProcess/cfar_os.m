function obj = cfar_os(obj)
    data_dopplerFFT = obj.data_dopplerFFT;
    sig_integrate = reshape(data_dopplerFFT, size(data_dopplerFFT, 1), size(data_dopplerFFT,2), []);
    sig_integrate = sum((abs(sig_integrate)).^2,3) + 1;
    obj.sig_integrate = sig_integrate;
    
    obj = range_cfar_os(obj);

    if obj.res_range_cfar.n_obj > 0
        obj = doppler_cfar_os_cyclicity(obj);

        n_obj = obj.res_doppler_cfar.n_obj;        
        range_index = zeros(n_obj, 1);
        range_obj = zeros(n_obj, 1);
        doppler_index = zeros(n_obj, 1);
        doppler_obj = zeros(n_obj, 1);
        % use the first noise estimate, apply to the second detection
        noise = zeros(n_obj, 1);
        bin_val = zeros(n_obj, size(data_dopplerFFT, 3), size(data_dopplerFFT, 4));
        intensity = zeros(n_obj, 1);
        snr = zeros(n_obj, 1);
        for i = 1:n_obj
            range_index(i) = obj.res_doppler_cfar.index_obj(i, 1) - 1;
            range_obj(i) = range_index(i) * obj.mode_infos.range_res_m;

            doppler_index(i) = obj.res_doppler_cfar.index_obj(i, 2) - 1;
            doppler_obj(i) = (doppler_index(i) - obj.dopplerFFT_size / 2) * obj.mode_infos.velocity_res_m_sec;

            index_noise = find( ...
                obj.res_doppler_cfar.index_obj(i, 1) == obj.res_range_cfar.index_obj(:, 1) & ...
                obj.res_doppler_cfar.index_obj(i, 2) == obj.res_range_cfar.index_obj(:, 2) ...
                );            
            noise(i) = obj.res_range_cfar.noise_obj(index_noise);
            
            bin_val(i, :, :) = data_dopplerFFT(range_index(i)+1, doppler_index(i)+1, :, :);

            intensity(i) = sum(abs(bin_val(i, :, :)).^2, "all");
            snr(i) = 10 * log10(intensity(i) / noise(i));
        end

        res_rdm_cfar = struct();
        res_rdm_cfar.n_obj = n_obj;
        res_rdm_cfar.range_index = range_index;
        res_rdm_cfar.range = range_obj;
        res_rdm_cfar.doppler_index = doppler_index;
        res_rdm_cfar.doppler = doppler_obj;
        res_rdm_cfar.doppler_index_origin = doppler_index;
        res_rdm_cfar.doppler_origin = doppler_obj;
        res_rdm_cfar.noise = noise;
        res_rdm_cfar.bin_val = bin_val;
        res_rdm_cfar.intensity = intensity;
        res_rdm_cfar.snr = snr;

        obj.res_rdm_cfar = res_rdm_cfar;
        obj = TDMA_phase_component(obj);

    else
        res_rdm_cfar = struct();
        obj.res_rdm_cfar = res_rdm_cfar;
    end

end

function obj = range_cfar_os(obj)
    refWinSize = 8;
    guardWinSize = 8;
    K0 = 5;
    discardCellLeft = 10;
    discardCellRight = 20;
    maxEnable = 0;
    sortSelectFactor = 0.75;
    gaptot = refWinSize + guardWinSize;
    n_obj = 0;
    index_obj = [];
    intensity_obj = [];
    noise_obj = [];
    snr_obj = [];

    n_range = size(obj.sig_integrate, 1);
    n_doppler = size(obj.sig_integrate, 2);
    for i_doppler = 1:n_doppler
        sigv = obj.sig_integrate(:, i_doppler);
        vecMid = sigv(discardCellLeft+1: n_range - discardCellRight);
        vecLeft = vecMid(1:gaptot);
        vecRight = vecMid(end-(gaptot)+1:end);
        vec = [vecLeft; vecMid; vecRight];

        for j = 1:size(vecMid, 1)
            index_cur = j + gaptot;
            % reference index
            index_left = (index_cur - gaptot):(index_cur - guardWinSize - 1);
            index_right = (index_cur + guardWinSize + 1):(index_cur + gaptot);

            sorted_res = sort(vec([index_left, index_right]));
            value_selected = sorted_res(ceil(sortSelectFactor*size(sorted_res, 1)));

            if maxEnable == 1
                % whether value_selected is the local max value
                value_local = vec(index_cur - gaptot: index_cur + gaptot);
                value_max = max(value_local);
                if vec(index_cur) >= K0 * value_selected && vec(index_cur) >= value_max
                    n_obj = n_obj + 1;
                    index_obj = [index_obj; [discardCellLeft + j, i_doppler]];
                    intensity_obj = [intensity_obj; vec(index_cur)];
                    noise_obj = [noise_obj; value_selected];
                    snr_obj = [snr_obj; vec(index_cur) / value_selected];
                end
            else
                if vec(index_cur) >= K0 * value_selected
                    n_obj = n_obj + 1;
                    index_obj = [index_obj; [discardCellLeft + j, i_doppler]];
                    intensity_obj = [intensity_obj; vec(index_cur)];
                    noise_obj = [noise_obj; value_selected];
                    snr_obj = [snr_obj; vec(index_cur) / value_selected];
                end
            end
        end
    end

    res_range_cfar = struct();
    res_range_cfar.n_obj = n_obj;
    res_range_cfar.index_obj = index_obj;
    res_range_cfar.intensity_obj = intensity_obj;
    res_range_cfar.noise_obj = noise_obj;
    res_range_cfar.snr_obj = snr_obj;
    obj.res_range_cfar = res_range_cfar;
end

function obj = doppler_cfar_os_cyclicity(obj)
    refWinSize = 4;
    guardWinSize = 0;
    K0 = 0.5;
    maxEnable = 0;
    sortSelectFactor = 0.75;
    gaptot = refWinSize + guardWinSize;
    n_obj = 0;
    index_obj = [];
    intensity_obj = [];
    noise_obj = [];
    snr_obj = [];

    index_obj_range = obj.res_range_cfar.index_obj;
    index_obj_range_unique = unique(index_obj_range(:, 1));

    for i = 1:size(index_obj_range_unique, 1)
        i_range = index_obj_range_unique(i);
        sigv = obj.sig_integrate(i_range, :);
        % cyclicity
        vecMid = sigv;
        vecLeft = sigv(end-gaptot+1:end);
        vecRight = sigv(1: gaptot);
        vec = [vecLeft, vecMid, vecRight];

        for j = 1:size(vecMid, 2)
            index_cur = j + gaptot;
            index_left = (index_cur - gaptot):(index_cur - guardWinSize - 1);
            index_right = (index_cur + guardWinSize + 1):(index_cur + gaptot);

            sorted_res = sort(vec([index_left, index_right]));
            value_selected = sorted_res(ceil(sortSelectFactor*size(sorted_res, 2)));

            if maxEnable == 1
                % whether value_selected is the local max value
                value_local = vec(index_cur - gaptot: index_cur + gaptot);
                value_max = max(value_local);
                if vec(index_cur) >= K0 * value_selected && vec(index_cur) >= value_max
                    if ismember(j, index_obj_range(index_obj_range(:, 1)==i_range, 2))
                        n_obj = n_obj + 1;
                        index_obj = [index_obj; [i_range, j]];
                        intensity_obj = [intensity_obj; vec(index_cur)];
                        noise_obj = [noise_obj; value_selected];
                        snr_obj = [snr_obj; vec(index_cur) / value_selected];
                    end
                end
            else
                if vec(index_cur) >= K0 * value_selected
                    if ismember(j, index_obj_range(index_obj_range(:, 1)==i_range, 2))
                        n_obj = n_obj + 1;
                        index_obj = [index_obj; [i_range, j]];
                        intensity_obj = [intensity_obj; vec(index_cur)];
                        noise_obj = [noise_obj; value_selected];
                        snr_obj = [snr_obj; vec(index_cur) / value_selected];
                    end
                end
            end
        end
    end

    res_doppler_cfar = struct();
    res_doppler_cfar.n_obj = n_obj;
    res_doppler_cfar.index_obj = index_obj;
    res_doppler_cfar.intensity_obj = intensity_obj;
    res_doppler_cfar.noise_obj = noise_obj;
    res_doppler_cfar.snr_obj = snr_obj;
    obj.res_doppler_cfar = res_doppler_cfar;
end