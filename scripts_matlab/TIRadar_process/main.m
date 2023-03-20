clearvars; close all; clc;
addpath("3rdpart\jsonlab");

root_group = 'F:\ourDataset_v2\20221217_group0000_mode1_279frames';
apply_vmax_extend = true;

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

    nmp = dopplerFFT(nmp);

    nmp = cfar_os(nmp);

    if ~isempty(nmp.res_doppler_cfar)
        nmp = doa(nmp);
        if ~isempty(nmp.res_doa)
            pcd = generate_pcd(nmp);
        end
    end

    

end