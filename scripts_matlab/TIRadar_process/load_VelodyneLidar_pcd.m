function pcd = load_VelodyneLidar_pcd(pcd_path)    
    data = readmatrix(pcd_path, 'FileType', 'delimitedtext');
    data = num2cell(data);
    pcd = cell2struct(data, {'x', 'y', 'z', 'intensity', 'idx_laser', 'unix_timestamp'}, 2);
end