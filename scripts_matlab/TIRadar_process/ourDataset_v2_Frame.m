classdef ourDataset_v2_Frame
    properties
        root = '';  
        frame_name = '';
        frame_id = -1;
    end

    methods
        function obj = ourDataset_v2_Frame(root_frame)
            obj.root = root_frame;

            tmp = split(obj.root, filesep);
            obj.frame_name = tmp{end};            
            obj.frame_id = str2double(replace(obj.frame_name, 'frame', ''));
        end

        function [data_raw, mode_infos, calibmat, ts_str] = get_TIRadar_data(obj)
            root_TIRadar = fullfile(obj.root, 'TIRadar');

            name_adcdata_npz = dir(fullfile(root_TIRadar, '*.adcdata.npz')).name;
            path_adcdata_npz = fullfile(root_TIRadar, name_adcdata_npz);
            [data_real, data_imag, mode_infos] = read_adcdata_npz(path_adcdata_npz);
            data_raw = double(data_real) + 1j * double(data_imag);
            
            name_calibmat = dir(fullfile(root_TIRadar, '*.mat')).name;
            path_calibmat = fullfile(root_TIRadar, name_calibmat);
            calibmat = load(path_calibmat);
            
            name_json = dir(fullfile(root_TIRadar, '*.json')).name;
            path_json = fullfile(root_TIRadar, name_json);
            data_json = loadjson(path_json);
            ts_str = data_json.timestamp;
        end

        function [image, ts_str] = get_LeopardCamera0_data(obj)
            root_LeopardCamera0 = fullfile(obj.root, 'LeopardCamera0');

            name_image_png = dir(fullfile(root_LeopardCamera0, '*.png')).name;
            path_image_png = fullfile(root_LeopardCamera0, name_image_png);
            image = imread(path_image_png);

            name_json = dir(fullfile(root_LeopardCamera0, '*.json')).name;
            path_json = fullfile(root_LeopardCamera0, name_json);
            data_json = loadjson(path_json);
            ts_str = data_json.timestamp;
        end

        function [pcd, ts_str] = get_VelodyneLidar_data(obj)
            root_VelodyneLidar = fullfile(obj.root, 'VelodyneLidar');

            name_pcd = dir(fullfile(root_VelodyneLidar, '*.pcd')).name;
            path_pcd = fullfile(root_VelodyneLidar, name_pcd);
            pcd = load_VelodyneLidar_pcd(path_pcd);

            name_json = dir(fullfile(root_VelodyneLidar, '*.json')).name;
            path_json = fullfile(root_VelodyneLidar, name_json);
            data_json = loadjson(path_json);
            ts_str = data_json.timestamp;
        end
    end

end