function obj = reorder(obj)    
    obj.data_reordered = obj.data_calib(:, :, obj.rx_id_onboard, :);
end