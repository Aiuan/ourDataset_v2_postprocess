function res = pcd_in_zone(pcd, xlim, ylim, zlim)
    x = [pcd.x];
    mask_x = (x>=xlim(1) & x<=xlim(2));

    y = [pcd.y];
    mask_y = (y>=ylim(1) & y<=ylim(2));

    z = [pcd.z];
    mask_z = (z>=zlim(1) & z<=zlim(2));

    mask = (mask_x & mask_y & mask_z);
    
    keys = fieldnames(pcd);

    data = zeros(sum(mask), size(keys, 1));    
    for i = 1:size(data, 2)
        key = keys{i};
        value = [pcd.(key)];
        data(:, i) = value(mask);
    end
    data = num2cell(data);
    res = cell2struct(data, keys, 2);

end