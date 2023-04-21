clearvars; close all; clc;
addpath("3rdpart\jsonlab");

root_group = 'F:\ourDataset_v2\20221217_group0000_mode1_279frames';
apply_vmax_extend = true;
vis = true;

%% range-doppler-azimuth-elevation
group = ourDataset_v2_Group(root_group);
for idx_frame = 1:group.num_frames
    disp("================================================================"); 
    fprintf("%s\n", group.group_name);

    frame = get_frame(group, idx_frame);
    fprintf("(%d/%d)%s\n", frame.frame_id + 1, group.num_frames, frame.frame_name);

    [data_raw, mode_infos, calibmat, ts_str] = get_TIRadar_data(frame);
    fprintf("TIRadar: unix_timestamp = %s\n", ts_str);

    nmp = NormalModeProcess(data_raw, mode_infos, calibmat, apply_vmax_extend);

    nmp = calibrate(nmp);

    nmp = reorder(nmp);

    nmp = calculate_antenna_array(nmp);

    nmp = rangeFFT(nmp);

    heatmap_bev = generate_heatmap_bev(nmp);

    nmp = dopplerFFT(nmp);

    nmp = cfar_os(nmp);

    if ~isempty(nmp.res_doppler_cfar)
        nmp = doa(nmp);
        if ~isempty(nmp.res_doa)
            pcd = generate_pcd(nmp);
        end
    end

    %% vis
    if vis
        sig_integrate_dB = 10 * log10(nmp.sig_integrate);
        lim_range = ceil(nmp.range_bins(end)/10)*10;

        % range-doppler curves
        subplot(2, 3, 1);
        v0_index = size(sig_integrate_dB, 2) / 2 + 1;
        plot(nmp.range_bins, sig_integrate_dB(:, v0_index), 'k', 'LineWidth', 4);
        hold on;
        for i = 1:size(sig_integrate_dB, 2)
            if i == v0_index
                color = [0, 0, 0];
            else
                color = project2jet(i, size(sig_integrate_dB, 2), 1);
            end
            plot(nmp.range_bins, sig_integrate_dB(:, i), 'Color', color);
            if ~isempty(nmp.res_rdm_cfar)
                i_doppler = i-1;
                mask = (nmp.res_rdm_cfar.doppler_index == i_doppler);
                i_range = nmp.res_rdm_cfar.range_index(mask);
                if sum(mask) > 0
                    plot(nmp.res_rdm_cfar.range(mask), ...
                        sig_integrate_dB(i_range+1, i_doppler+1), ...
                        'o', 'LineWidth', 2, 'MarkerEdgeColor', color, 'MarkerSize',6);
                end
            end

        end
        hold off;
        grid on;
        xlabel('range(m)');
        ylabel('intensity(dB)');
        title('range-doppler curves');

        % range-doppler map
        subplot(2,3,2);
        imagesc(nmp.range_bins, nmp.doppler_bins, sig_integrate_dB);
        c = colorbar();
        c.Label.String = 'intensity(dB)';
        colormap('jet');
        xlabel('range(m)');
        ylabel('doppler(m/s)');
        title('range-doppler map');

        % heatmap_bev
        subplot(2, 3, 3);
        surf(heatmap_bev.x, heatmap_bev.y, heatmap_bev.value, 'EdgeColor', 'none');
        view([0 90]);
        clim([min(heatmap_bev.value, [], 'all'), max(heatmap_bev.value, [], 'all')]);
        c = colorbar;
        c.Label.String = 'intensity';
        colormap('jet');
        xlabel('x(m)');
        ylabel('y(m)');
        title('heatmap_bev', 'Interpreter', 'none');
        
        % pcd
        subplot(2, 3, 4);
        scatter3([pcd.x], [pcd.y], [pcd.z], [pcd.snr], [pcd.doppler], 'filled');        
        clim([nmp.doppler_bins(1), nmp.doppler_bins(end)]);
        grid on;
        axis('image');        
        xlim([-lim_range/2, lim_range/2]);
        ylim([0, lim_range]);
        c = colorbar;
        c.Label.String = 'doppler(m/s)';
        colormap('jet');
        view([0 90]); 
        xlabel('x(m)');
        ylabel('y(m)');
        zlabel('z(m)');
        title('radar pcd');

        % lidar
        [pcd_VelodyneLidar, ts_str_VelodyneLidar] = get_VelodyneLidar_data(frame);
        pcd_VelodyneLidar = pcd_in_zone(pcd_VelodyneLidar, [-lim_range/2, lim_range/2], [0, lim_range], [-inf, inf]);
        
        subplot(2, 3, 5);
        scatter3([pcd_VelodyneLidar.x], [pcd_VelodyneLidar.y], [pcd_VelodyneLidar.z], ...
            4, [pcd_VelodyneLidar.intensity], 'filled');        
        clim([0, 255]);
        grid on;
        axis('image');        
        xlim([-lim_range/2, lim_range/2]);
        ylim([0, lim_range]);
        c = colorbar;
        c.Label.String = 'intensity';
        colormap('jet');
        view([0 90]); 
        xlabel('x(m)');
        ylabel('y(m)');
        zlabel('z(m)');
        title('lidar pcd');


        % image
        [image_LeopardCamera0, ts_str_LeopardCamera0] = get_LeopardCamera0_data(frame);
        subplot(2, 3, 6);
        imshow(image_LeopardCamera0);
        title('LeopardCamera0');
    end

    

end