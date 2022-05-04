import numpy as np


def process_range_data(data, k):
    # data: LaserScan
    # k: window size for mean-filtering data.ranges
    ranges = np.array(data.ranges)
    is_valid = np.logical_and(data.range_min <= ranges, ranges <= data.range_max)
    if not np.any(is_valid):
        return None
    ranges[np.logical_not(is_valid)] = 0
    cs = np.cumsum(ranges)
    n_cv = np.cumsum(is_valid)
    # mean-filter the ranges to make it more robust
    avg_ranges = np.empty_like(ranges)
    for i in range(360):
        l = i - k // 2
        r = i + (k + 1) // 2
        if l < 0:
            n_valid = n_cv[-1] - n_cv[l + 359] + n_cv[r - 1]
            sum_range = cs[-1] - cs[l + 359] + cs[r - 1]
        elif r > 360:
            n_valid = n_cv[-1] - n_cv[l - 1] + n_cv[r - 361]
            sum_range = cs[-1] - cs[l - 1] + cs[r - 361]
        else:
            n_valid = n_cv[r - 1] - (n_cv[l - 1] if l else 0)
            sum_range = cs[r - 1] - (cs[l - 1] if l else 0)
        avg_ranges[i] = sum_range / n_valid if n_valid else np.nan
    # get the degree and distance of the closest point
    deg = np.nanargmin(avg_ranges)
    dist = avg_ranges[deg]
    if deg >= 180:
        deg -= 360
    return deg, dist
