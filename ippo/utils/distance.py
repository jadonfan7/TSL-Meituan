import math
import numpy as np

def manhattan(p1:tuple, p2:tuple) -> float: #  L1 distanc
    dist = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    return float(dist)

def haversine(loc1, loc2):
    lat1, lon1, lat2, lon2 = loc1[1], loc1[0], loc2[1], loc2[0]
    lat1 = lat1/10**6
    lat2 = lat2/10**6
    lon1 = lon1/10**6
    lon2 = lon2/10**6

    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance*1000

if __name__ == '__main__':
    print(manhattan((0,0), (7.5,9.9)))
    print(manhattan((0,0),(7,9)))
    if math.isclose(manhattan((0,0),(7.5,9.9)), 17.4):
        print('equal to 17.4')
    if math.isclose(manhattan((0,0),(7,9.0)), 16):
        print('equal to 16')
