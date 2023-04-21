function pcd = generate_pcd(obj)
    range = obj.res_doa.range;
    azimuth = obj.res_doa.azimuth;
    elevation = obj.res_doa.elevation;
    x = range .* cosd(elevation) .* sind(azimuth);
    y = range .* cosd(elevation) .* cosd(azimuth);
    z = range .* sind(elevation);
    doppler = obj.res_doa.doppler;
    snr = obj.res_doa.snr;
    intensity = obj.res_doa.intensity;
    noise = obj.res_doa.noise;

    pcd = struct( ...
        'x', num2cell(x), ...
        'y', num2cell(y), ...
        'z', num2cell(z), ...
        'doppler', num2cell(doppler), ...
        'snr', num2cell(snr), ...
        'intensity', num2cell(intensity), ...
        'noise', num2cell(noise), ...
        'range', num2cell(range), ...
        'azimuth', num2cell(azimuth), ...
        'elevation', num2cell(elevation));

end