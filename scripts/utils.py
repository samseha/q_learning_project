def process_range_data(data, lo=-180, hi=180):
    # returns the angle and distance of the closest object within the angular range [lo, hi)
    # data: LaserScan
    min_dist = None
    for i in range(360):
        if i < hi or i - 360 >= lo:
            if data.range_min < data.ranges[i] < data.range_max:
                if min_dist is None or data.ranges[i] < min_dist:
                    ang = i
                    min_dist = data.ranges[i]
    if min_dist is None:
        return None
    if ang >= 180:
        ang -= 360
    return ang, min_dist
