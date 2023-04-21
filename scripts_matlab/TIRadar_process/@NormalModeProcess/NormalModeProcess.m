classdef NormalModeProcess
    properties        
        data_raw;
        mode_infos;
        rangeFFT_size;
        dopplerFFT_size;
        azimuthFFT_size = 256;
        elevationFFT_size = 256;
        TI_Cascade_Antenna_DesignFreq_GHz = 76.8;
        doa_unitDis;
        range_bins;
        doppler_bins;
        azimuth_bins;
        elevation_bins;
        calibmat;
        num_rx;
        num_tx;
        rx_id;
        rx_id_onboard = [13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8];
        rx_position_azimuth = [11, 12, 13, 14, 50, 51, 52, 53, 46, 47, 48, 49, 0, 1, 2, 3];
        rx_position_elevation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        tx_id_transfer_order = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        tx_id;
        tx_id_onboard = [12, 11, 10, 3, 2, 1, 9, 8, 7, 6, 5, 4];
        tx_position_azimuth = [11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0];
        tx_position_elevation = [6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        apply_vmax_extend;
        min_dis_apply_vmax_extend = 10;

        data_calib;
        data_reordered;

        % calculate_antenna_array
        virtual_array_azimuth;
        virtual_array_elevation;
        virtual_array_tx_id;
        virtual_array_rx_id;
        virtual_array;
        virtual_array_index_noredundant;
        virtual_array_noredundant;
        virtual_array_index_redundant;
        virtual_array_redundant;
        virtual_array_info_overlaped_diff1tx;
        virtual_array_noredundant_row1;
        
        % rangeFFT
        scale_factor = [0.2500, 0.1250, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039, 0.0020];
        rangeFFT_window_enable=true;
        rangeFFT_scale_on=false;
        data_rangeFFT;
        
        % dopplerFFT
        dopplerFFT_window_enable=false;
        dopplerFFT_clutter_remove=false;
        dopplerFFT_scale_on=false;        
        data_dopplerFFT;

        % cfar_os
        sig_integrate;
        res_range_cfar;
        res_doppler_cfar;
        res_rdm_cfar;
        
        % doa
        res_doa;

    end

    methods
        function obj = NormalModeProcess(data_raw, mode_infos, calibmat, apply_vmax_extend)
            obj.data_raw = data_raw;
            obj.mode_infos = mode_infos;
            
            obj.rangeFFT_size = size(data_raw, 1);
            obj.dopplerFFT_size = size(data_raw, 2);
            
            obj.doa_unitDis = 0.5 * obj.mode_infos.freq_center_GHz / ...
                obj.TI_Cascade_Antenna_DesignFreq_GHz;
            
            obj.range_bins = (0: obj.rangeFFT_size-1) * obj.mode_infos.range_res_m;
            obj.doppler_bins = (-obj.dopplerFFT_size/2: obj.dopplerFFT_size/2-1) * ...
                obj.mode_infos.velocity_res_m_sec;
            obj.azimuth_bins = (-obj.azimuthFFT_size:2:obj.azimuthFFT_size-2) * ...
                pi / obj.azimuthFFT_size;
            obj.elevation_bins = (-obj.elevationFFT_size:2:obj.elevationFFT_size-2) * ...
                pi / obj.elevationFFT_size;

            obj.calibmat = calibmat;

            obj.num_rx = obj.mode_infos.numRXPerDevice * obj.mode_infos.numDevices;
            obj.num_tx = obj.mode_infos.numTXPerDevice * obj.mode_infos.numDevices;

            obj.rx_id = 1:obj.num_rx;
            obj.tx_id = 0:obj.num_tx;

            obj.apply_vmax_extend = apply_vmax_extend;
        end
        
       obj = calibrate(obj);

       obj = reoder(obj);

       obj = calculate_antenna_array(obj);

       obj = rangeFFT(obj);

       obj = dopplerFFT(obj);

       obj = cfar_os(obj);

       obj = TDMA_phase_component(obj);

       obj = doa(obj);

       pcd = generate_pcd(obj);

       heatmap_bev = generate_heatmap_bev(obj);
    end
end