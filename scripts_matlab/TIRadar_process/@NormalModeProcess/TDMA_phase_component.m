function obj = TDMA_phase_component(obj)

    if obj.apply_vmax_extend
        doppler_index = obj.res_rdm_cfar.doppler_index;
        dopplerFFT_size = obj.dopplerFFT_size;
        doppler_obj = obj.res_rdm_cfar.doppler;
        num_tx = double(obj.num_tx);
        doppler_index_unwrap = doppler_index + dopplerFFT_size * ...
            ((0:num_tx-1) - num_tx / 2 + 1 * (doppler_obj <= 0));

        sig_bin_org = obj.res_rdm_cfar.bin_val;

        % Doppler phase correction due to TDM MIMO
        delta_phi = 2 * pi * (doppler_index_unwrap - dopplerFFT_size / 2) ./ (num_tx * dopplerFFT_size);
        
        % construct all possible signal vectors based on the number of possible hypothesis
        correct_matrix = exp(-1j .* ...
            reshape((0:num_tx-1), 1, 1, [], 1) .* ...
            reshape(delta_phi, size(delta_phi,1), 1, 1, size(delta_phi,2)) ...
            );
        sig_bin = reshape(sig_bin_org, size(sig_bin_org, 1), size(sig_bin_org, 2), size(sig_bin_org, 3), 1) .* correct_matrix;
        
        % use overlap antenna to do max doppler unwrap
        virtual_array_info_overlaped_diff1tx = obj.virtual_array_info_overlaped_diff1tx;
        rx_id_onboard = obj.rx_id_onboard;
        tx_id_transfer_order = obj.tx_id_transfer_order;
        index_antenna_overlaped_diff1tx = zeros(size(virtual_array_info_overlaped_diff1tx, 1), 4);
        for i = 1:size(virtual_array_info_overlaped_diff1tx, 1)
            index_antenna_overlaped_diff1tx(i, 1) = find(rx_id_onboard == virtual_array_info_overlaped_diff1tx(i, 3));
            index_antenna_overlaped_diff1tx(i, 2) = find(tx_id_transfer_order == virtual_array_info_overlaped_diff1tx(i, 4));
            index_antenna_overlaped_diff1tx(i, 3) = find(rx_id_onboard == virtual_array_info_overlaped_diff1tx(i, 7));
            index_antenna_overlaped_diff1tx(i, 4) = find(tx_id_transfer_order == virtual_array_info_overlaped_diff1tx(i, 8));
        end
        
        sig_overlap = zeros(size(sig_bin_org, 1), size(index_antenna_overlaped_diff1tx, 1), 2);
        for i = 1:size(sig_overlap, 2)
            sig_overlap(:, i, 1) = sig_bin_org(:, index_antenna_overlaped_diff1tx(i, 1), index_antenna_overlaped_diff1tx(i, 2));
            sig_overlap(:, i, 2) = sig_bin_org(:, index_antenna_overlaped_diff1tx(i, 3), index_antenna_overlaped_diff1tx(i, 4));
        end

        % check the phase difference of each overlap antenna pair for each hypothesis
        angle_sum_test = zeros(size(sig_overlap, 1), size(sig_overlap, 2), size(delta_phi, 2));
        for i = 1:size(angle_sum_test, 2)
            tmp1 = sig_overlap(:, 1:i, 2);
            tmp1 = reshape(tmp1.', size(tmp1, 2), 1, size(tmp1, 1));
            tmp2 = exp(-1j * delta_phi);
            tmp2 = reshape(tmp2.', 1, size(tmp2, 2), size(tmp2, 1));
            signal2 = pagemtimes(tmp1, tmp2);
            signal2 = permute(signal2, [3, 1, 2]);
            
            tmp3 = sig_overlap(:, 1:i, 1);
            tmp3 = reshape(tmp3, size(tmp3, 1), size(tmp3, 2), 1);
            angle_sum_test(:, i, :) = angle(squeeze(sum(tmp3 .* conj(signal2), 2)));
        end

        % chosee the hypothesis with minimum phase difference to estimate the unwrap factor
        [~, doppler_unwrap_integ_overlap_index] = min(abs(angle_sum_test), [], 3);
        
        % test the angle FFT SNR
        virtual_array_noredundant_row1 = obj.virtual_array_noredundant_row1;
        index_antenna_noredundant_row1 = zeros(size(virtual_array_noredundant_row1, 1), 2);
        for i = 1:size(index_antenna_noredundant_row1, 1)
            index_antenna_noredundant_row1(i, 1) = find(rx_id_onboard == virtual_array_noredundant_row1(i, 3));
            index_antenna_noredundant_row1(i, 2) = find(tx_id_transfer_order == virtual_array_noredundant_row1(i, 4));
        end
        sig_bin_row1 = zeros(size(sig_bin, 1), size(index_antenna_noredundant_row1, 1), size(sig_bin, 4));
        for i = 1:size(sig_bin_row1, 2)
            sig_bin_row1(:, i, :) = sig_bin(:, index_antenna_noredundant_row1(i, 1), index_antenna_noredundant_row1(i, 2), :);
        end
        angleFFT_size = 128;
        sig_bin_row1_fft = fftshift(fft(sig_bin_row1, angleFFT_size, 2), 2);
        angle_bin_skip_left = 4;
        angle_bin_skip_right = 4;
        sig_bin_row1_fft_cut = abs(sig_bin_row1_fft(:, angle_bin_skip_left+1:angleFFT_size-angle_bin_skip_right, :));
        [~, doppler_unwrap_integ_FFT_index] = max(squeeze(max(sig_bin_row1_fft_cut, [], 2)), [], 2);
        
        doppler_unwrap_integ_index = zeros(size(doppler_unwrap_integ_overlap_index, 1), 1);
        for i = 1:size(doppler_unwrap_integ_index, 1)
            values = unique(doppler_unwrap_integ_overlap_index(i, :));
            if size(values, 2) > 1
                counts = hist(doppler_unwrap_integ_overlap_index(i, :), values);
                [~, idx] = max(counts);
                doppler_unwrap_integ_index(i, 1) = values(idx);
            else
                doppler_unwrap_integ_index(i, 1) = values;
            end
        end
        
        % corret doppler after applying the integer value
        bin_val_correct = zeros(size(sig_bin_org));
        doppler_index_correct = zeros(size(doppler_index));
        doppler_index_correct_FFT = zeros(size(doppler_index));
        doppler_index_correct_overlap = zeros(size(doppler_index));
        for i = 1:size(bin_val_correct, 1)
            bin_val_correct(i, :, :) = sig_bin(i, :, :, doppler_unwrap_integ_index(i));
            doppler_index_correct(i) = doppler_index_unwrap(i, doppler_unwrap_integ_index(i));
            doppler_index_correct_FFT(i) = doppler_index_unwrap(i, doppler_unwrap_integ_FFT_index(i));
            doppler_index_correct_overlap(i) = doppler_index_unwrap(i, doppler_unwrap_integ_index(i));
        end
        
        obj.res_rdm_cfar.bin_val_correct = bin_val_correct;
        obj.res_rdm_cfar.doppler_index_correct = doppler_index_correct;
        obj.res_rdm_cfar.doppler_correct = (doppler_index_correct - dopplerFFT_size / 2) * obj.mode_infos.velocity_res_m_sec;
        obj.res_rdm_cfar.doppler_index_correct_FFT = doppler_index_correct_FFT;
        obj.res_rdm_cfar.doppler_correct_FFT = (doppler_index_correct_FFT - dopplerFFT_size / 2) * obj.mode_infos.velocity_res_m_sec;
        obj.res_rdm_cfar.doppler_index_correct_overlap = doppler_index_correct_overlap;
        obj.res_rdm_cfar.doppler_correct_overlap = (doppler_index_correct_overlap - dopplerFFT_size / 2) * obj.mode_infos.velocity_res_m_sec;
        
        % replace doppler_index: doppler_index_correct, doppler: doppler_correct
        obj.res_rdm_cfar.doppler_index = obj.res_rdm_cfar.doppler_index_correct;
        obj.res_rdm_cfar.doppler = obj.res_rdm_cfar.doppler_correct;
        
        % apply min_dis_apply_vmax_extend
        index_noapply = find(obj.res_rdm_cfar.range <= obj.min_dis_apply_vmax_extend);
        
        % doppler phase correction due to TDM MIMO without applying vmax extention algorithm
        delta_phi_noapply = 2 * pi * (doppler_index - size(obj.data_dopplerFFT, 2) / 2) / ...
            (size(obj.data_dopplerFFT, 4) * size(obj.data_dopplerFFT, 2));
        i_tx_noapply = 0:double(obj.num_tx)-1;
        sig_bin_noapply = sig_bin_org .* reshape(exp(-1j .* delta_phi_noapply * i_tx_noapply), size(delta_phi_noapply, 1), 1, size(i_tx_noapply, 2));
        
        obj.res_rdm_cfar.bin_val_correct(index_noapply, :, :) = sig_bin_noapply(index_noapply, :, :);
        obj.res_rdm_cfar.doppler_index_correct(index_noapply) = obj.res_rdm_cfar.doppler_index_origin(index_noapply);
        obj.res_rdm_cfar.doppler_correct(index_noapply) = obj.res_rdm_cfar.doppler_origin(index_noapply);
        obj.res_rdm_cfar.doppler_index_correct_FFT(index_noapply) = obj.res_rdm_cfar.doppler_index_origin(index_noapply);
        obj.res_rdm_cfar.doppler_correct_FFT(index_noapply) = obj.res_rdm_cfar.doppler_origin(index_noapply);
        obj.res_rdm_cfar.doppler_index_correct_overlap(index_noapply) = obj.res_rdm_cfar.doppler_index_origin(index_noapply);
        obj.res_rdm_cfar.doppler_correct_overlap(index_noapply) = obj.res_rdm_cfar.doppler_origin(index_noapply);
        
    else
        sig_bin_org = obj.res_rdm_cfar.bin_val;
        
        doppler_index = obj.res_rdm_cfar.doppler_index;        
        delta_phi = 2 * pi * (doppler_index - size(obj.data_dopplerFFT, 2) / 2) / ...
            (size(obj.data_dopplerFFT, 4) * size(obj.data_dopplerFFT, 2));

        i_tx = 0:size(obj.data_dopplerFFT, 4)-1;

        correct_matrix = exp(-1j * delta_phi * i_tx);
        correct_matrix = reshape(correct_matrix, size(delta_phi, 1), 1, size(i_tx, 2));

        sig_bin_correct = sig_bin_org .* correct_matrix;
        obj.res_rdm_cfar.bin_val_correct = sig_bin_correct;

    end
end