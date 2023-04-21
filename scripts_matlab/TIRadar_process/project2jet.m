function rgb = project2jet(value, vmax, vmin)
    n = size(value, 1);
    
    value_clamp = min(max(value, vmin), vmax);
    value_norm01 = (value_clamp - vmin) / (vmax - vmin);
    
    rgb = zeros(n, 3);
    rgb(:, 1) = red(value_norm01);
    rgb(:, 2)  = green(value_norm01);
    rgb(:, 3)  = blue(value_norm01);
end

function res = interpolate(value, y0, x0, y1, x1)
    res = (value - x0) * (y1 - y0) / (x1 - x0) + y0;
end

function res = base(value)
    if value > -0.75 && value <= -0.25
        res = interpolate(value, 0, -0.75, 1, -0.25);
    elseif value > -0.25 && value <= 0.25
        res = 1.0;
    elseif value > 0.25 && value <= 0.75
        res = interpolate(value, 1, 0.25, 0, 0.75);
    else
        res = 0;
    end
end

function res = red(value_norm01)
    value = value_norm01 * 2 - 1;
    res = base(value - 0.5);
end

function res = green(value_norm01)
    value = value_norm01 * 2 - 1;
    res = base(value);
end

function res = blue(value_norm01)
    value = value_norm01 * 2 - 1;
    res = base(value + 0.5);
end