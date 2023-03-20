function [data_real, data_imag, mode_infos] = read_adcdata_npz(adcdata_npz_path)
    [data_real, data_imag, mode_infos] = pyrun( ...
        [ ...
            "import numpy as np", ...
            "import array", ...
            "data = np.load(path, allow_pickle=True)", ...
            "data_real = array.array('i', data['data_real'].flatten())", ...
            "data_imag = array.array('i', data['data_imag'].flatten())", ...
            "mode_infos = data['mode_infos'][()]", ...
            "res = {'data_real': data_real, 'data_imag': data_imag, 'mode_infos': mode_infos}"
        ], ...
        ["data_real", "data_imag", "mode_infos"], ...
        path=adcdata_npz_path ...
    );

    mode_infos = struct(mode_infos);
    mode_infos.chirpEndIdx = int32(mode_infos.chirpEndIdx);
    mode_infos.chirpStartIdx = int32(mode_infos.chirpStartIdx);
    mode_infos.numAdcSamples = int32(mode_infos.numAdcSamples);
    mode_infos.numChirps = int32(mode_infos.numChirps);
    mode_infos.numDevices = int32(mode_infos.numDevices);
    mode_infos.numFrames = int32(mode_infos.numFrames);
    mode_infos.numLoops = int32(mode_infos.numLoops);
    mode_infos.numRXPerDevice = int32(mode_infos.numRXPerDevice);
    mode_infos.numTXPerDevice = int32(mode_infos.numTXPerDevice);

    mode_infos.name = char(mode_infos.name);

    n1 = mode_infos.numAdcSamples;
    n2 = mode_infos.numLoops;
    n3 = mode_infos.numRXPerDevice * mode_infos.numDevices;
    n4 = mode_infos.numTXPerDevice * mode_infos.numDevices;

    data_real = int16(data_real);
    data_real = reshape(data_real, n4, n3, n2, n1);
    data_real = permute(data_real, [4,3,2,1]);

    data_imag = int16(data_imag);
    data_imag = reshape(data_imag, n4, n3, n2, n1);
    data_imag = permute(data_imag, [4,3,2,1]);

end