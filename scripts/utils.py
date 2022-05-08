def process_range_data(data, lo=-180, hi=180):
    # data: LaserScan
    min_dist = None
    for i in range(360):
        if i < hi or i - 360 >= lo:
            if data.range_min < data.ranges[i] < data.range_max:
                if min_dist is None or data.ranges[i] < min_dist:
                    deg = i
                    min_dist = data.ranges[i]
    if min_dist is None:
        return None
    if deg >= 180:
        deg -= 360
    return deg, min_dist
