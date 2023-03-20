function pcd = generate_pcd(obj)
    res_doa = obj.res_doa;
    pcd = struct();
    for i = 1:res_doa.n_obj
        pcd(i).range = obj.res_doa.range(i);
        pcd(i).azimuth = obj.res_doa.azimuth(i);
        pcd(i).elevation = obj.res_doa.elevation(i);

        pcd(i).x = pcd(i).range * cosd(pcd(i).elevation) * sind(pcd(i).azimuth);
        pcd(i).y = pcd(i).range * cosd(pcd(i).elevation) * cosd(pcd(i).azimuth);
        pcd(i).z = pcd(i).range * sind(pcd(i).elevation);

        pcd(i).doppler = obj.res_doa.doppler(i);
        pcd(i).snr = obj.res_doa.snr(i);
        pcd(i).intensity = obj.res_doa.intensity(i);
        pcd(i).noise = obj.res_doa.noise(i);
    end
end