function obj = calculate_antenna_array(obj)   
    virtual_array_azimuth = obj.tx_position_azimuth(obj.tx_id_transfer_order) + ...
        obj.rx_position_azimuth(obj.rx_id_onboard).';
    virtual_array_elevation = obj.tx_position_elevation(obj.tx_id_transfer_order) + ...
        obj.rx_position_elevation(obj.rx_id_onboard).';
    virtual_array_tx_id = repmat(obj.tx_id_transfer_order, obj.num_rx, 1);
    virtual_array_rx_id = repmat(obj.rx_id_onboard.', 1, obj.num_tx);

    % azimuth, elevation, rx_id, tx_id
    virtual_array = [
        reshape(virtual_array_azimuth, [], 1),...
        reshape(virtual_array_elevation, [], 1),...
        reshape(virtual_array_rx_id, [], 1),...
        reshape(virtual_array_tx_id, [], 1)
        ];

    % get antenna_noredundant
    [~, virtual_array_index_noredundant, ~] = unique(virtual_array(:, 1:2), "rows");
    virtual_array_noredundant = virtual_array(virtual_array_index_noredundant, :);

    % get antenna_redundant
    virtual_array_index_redundant = setxor((1:size(virtual_array, 1)), virtual_array_index_noredundant);
    virtual_array_redundant = virtual_array(virtual_array_index_redundant, :);

    % find and associate overlaped rx_tx pairs
    % azimuth, elevation, rx_associated, tx_associated, azimuth, elevation, rx_overlaped, tx_overlaped
    virtual_array_info_overlaped_associate = zeros(size(virtual_array_redundant, 1), 8);
    for i = 1:size(virtual_array_redundant, 1)
        mask = virtual_array_noredundant == virtual_array_redundant(i, :);
        mask = mask(:, 1) & mask(:, 2);
        info_associate = virtual_array_noredundant(mask, :);
        info_overlaped = virtual_array_redundant(i, :);
        virtual_array_info_overlaped_associate(i, :) = [info_associate, info_overlaped];
    end

    diff_tx = abs(virtual_array_info_overlaped_associate(:, 8) - virtual_array_info_overlaped_associate(:, 4));
    virtual_array_info_overlaped_diff1tx = virtual_array_info_overlaped_associate(diff_tx==1, :);
    
    [~, sorted_index] = sort(virtual_array_info_overlaped_diff1tx(:, 1));
    virtual_array_info_overlaped_diff1tx = virtual_array_info_overlaped_diff1tx(sorted_index, :);
    
    % find noredundant row1
    virtual_array_noredundant_row1 = virtual_array_noredundant(virtual_array_noredundant(:, 2)==0, :);
    
    
    obj.virtual_array_azimuth = virtual_array_azimuth;
    obj.virtual_array_elevation = virtual_array_elevation;
    obj.virtual_array_tx_id = virtual_array_tx_id;
    obj.virtual_array_rx_id = virtual_array_rx_id;
    obj.virtual_array = virtual_array;
    obj.virtual_array_index_noredundant = virtual_array_index_noredundant;
    obj.virtual_array_noredundant = virtual_array_noredundant;
    obj.virtual_array_index_redundant = virtual_array_index_redundant;
    obj.virtual_array_redundant = virtual_array_redundant;
    obj.virtual_array_info_overlaped_diff1tx = virtual_array_info_overlaped_diff1tx;
    obj.virtual_array_noredundant_row1 = virtual_array_noredundant_row1;
end