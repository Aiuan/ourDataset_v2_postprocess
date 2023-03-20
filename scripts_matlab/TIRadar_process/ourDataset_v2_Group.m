classdef ourDataset_v2_Group
    properties
        root = '';
        group_name = '';
        day = '';
        group_id = -1;
        mode = '';
        num_frames = -1;
        frame_names = {};
    end

    methods
        %% constructor
        function obj = ourDataset_v2_Group(root_group)
            obj.root = root_group;
            
            tmp = split(obj.root, filesep);
            obj.group_name = tmp{end};

            tmp = split(obj.group_name, '_');
            obj.day = tmp{1};
            obj.group_id = str2double(replace(tmp{2}, 'group', ''));
            obj.mode = tmp{3};

            items = dir(obj.root);
            items = sort_nat({items.name});
            obj.frame_names = items(3:end)';

            obj.num_frames = size(obj.frame_names, 1);
        end

        function frame = get_frame(obj, idx)
            root_frame = fullfile(obj.root, obj.frame_names{idx});
            frame = ourDataset_v2_Frame(root_frame);
        end
    end
end