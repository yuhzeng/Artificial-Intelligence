#!/usr/bin/env python3
"""
check if the gps in city-gps.txt makes sense

Hai Hu
"""

from route import MyMap

def check(mymap):

    cities_wrong_gps_int = {}  # { cityname : numOfOccurrences }

    dist_dict = {}  # { (cityInt1, cityInt2) : {'road_dist': 10, 'crowFlyDist': 7.5, 'ratio': 3/4} }

    # for each two cities in road-segment, give the road mile, and crowFlyDis
    for city1Int in mymap.MAP.keys():
        for city2Int in mymap.MAP[city1Int].keys():
            road_dist = mymap.MAP[city1Int][city2Int]['dist']
            # print('\ncity1 {} city2 {}: {}'.format(mymap.int2city[city1Int],
            #                                        mymap.int2city[city2Int],
            #                                        road_dist))
            crowFlyDist = None
            try:
                crowFlyDist = mymap.latLon2Miles(mymap.city2Coords[city1Int],
                                         mymap.city2Coords[city2Int])
                # print(crowFlyDist)
            except KeyError:
                pass
                # print('not in gps')

            # fill dict
            if crowFlyDist:
                #####  important  #####
                ratio = road_dist / crowFlyDist
                #####  important  #####
                x = min(city1Int, city2Int)
                y = max(city1Int, city2Int)
                dist_dict[ (x, y) ] = { 'road_dist': road_dist,
                                            'crowFlyDist': crowFlyDist,
                                            'ratio': ratio }

    counter = 0
    for cityTuple, d in reversed(sorted(dist_dict.items(), key=lambda x : x[1]['ratio'])):
        ratio = dist_dict[cityTuple]['ratio']
        if ratio < 1 or ratio > 20:  # wrong gps
            counter += 1

            cities_wrong_gps_int[cityTuple[0]] = cities_wrong_gps_int.get(cityTuple[0], 0) + 1
            cities_wrong_gps_int[cityTuple[1]] = cities_wrong_gps_int.get(cityTuple[1], 0) + 1

            city1Str = mymap.int2city[cityTuple[0]]
            city2Str = mymap.int2city[cityTuple[1]]

            road_dist = dist_dict[cityTuple]['road_dist']
            crowFlyDist = dist_dict[cityTuple]['crowFlyDist']

            print('ratio {:.5f}: road_dist {:3} < crowFlyDist {:.1f}: {} {}' \
                  .format(ratio, road_dist, crowFlyDist, city1Str, city2Str))
    print('\nwrong gps pairs:', counter)

    # for city, count in sorted(cities_wrong_gps_int.items(), key=lambda x: x[1]):
    #     print(count, city)
    #
    print('\nall cities:', len(mymap.city2int))
    print('\nnum of cities with wrong gps:', len(cities_wrong_gps_int))

    return set(cities_wrong_gps_int.keys())

if __name__ == '__main__':
    mymap = MyMap()
    cities_wrong_gps_int = check(mymap)
    # print(cities_wrong_gps_int)






