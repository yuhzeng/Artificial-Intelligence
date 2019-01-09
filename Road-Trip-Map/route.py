#!/usr/bin/env python3
"""
road trip problem, B551 hw1 part 2

team members: Hai Hu, Yuhan Zeng, Surya Prateek Soni

--------------------------------------------------------
Our heuristics for astar:

A. distance:

h1() = crow-fly distance. 

We tried the crow-fly distance from the current city to end city, calculated
using gps of the cities, using the formula from
http://www.movable-type.co.uk/scripts/latlong.html.
The crow-fly distance is the shortest distance between two points on the surface
of a sphere. So this should be admissible as long as the gps are correct. This
can reduce the search space drastically (sometimes several magnitude).

h2() = shortest neighbor.

However, since there are many problems with the gps, and it's very hard to clean them
thoroughly, we tried another heuristics. That is the shortest distance to any
neighbors, which should also be admissible since this distance must be less thatn
the distance to the end city. This is not as efficient, but can still reduce
the search space to about 3/4 of uniform search.

** in the current implementation, h2() is used, but we can easily change to h1(). **

B. time: the shortest time to get to any of the neighbors.

C. segments: segments of the current solution + 1, which is essentially the same
as the uniform search.
--------------------------------------------------------

In this script, we define 4 classes.

1. MyMap:

1.1 Basic structure

An object of this class represents the map we read in from 'road-segments.txt'.
The underlying structure is a dictionary, where the key is every *city* in the
text file, and the value is another dictionary that contains information about
the distance, time and speed limit for all its *neighbors*.

To speed up things, each city is represented by an int. And we use two maps to
keep track of the mapping from city_str to city_int and vice versa.

1.2 How we handle noisy data

A. road-segments.txt:

There are 3 possible noises: 1) The speed limit is unknown. 2) The speed limit
is zero. 3) More than 1 route between two cities. For the first two cases, we
simply ignore the two cities, as suggested in Piazza. There is only once case
for 3), where we just select the route with shortest distance.

B. city-gps.txt:

This is more problematic. We initially use this to calculate the *crow-fly* distance
between two cities, which is the shortest distance between two points on a sphere.
Then it turns out that for a huge number of city pairs, the *crow-fly* distance is
even larger than the road distance given in 'road-segments.txt' which should never
happen. We wrote a method check() to calculate a ratio for each city pair:

               road distance in 'road-segments.txt'
    ratio = -------------------------------------------
                crow-fly distance (based on gps)

The ratio should clearly be >= 1 since crow-fly distance is the shortest among two cities.
If ratio < 1, then must be something wrong with the gps, or the road-distance. Here
we assume it's the problem with the gps. Also, if the ratio is too high, there is
chance that the gps is wrong, since the road distance is too long compared to the
geographical distance.

If you run checkGPS.py, it will print out the sorted ratios, where ratio < 1 or ratio > 20.
There are more than 1000 pairs and about 2000 cities with suspicious gps! And that's
out of a total of 6527 cities. That's a lot if we simply say that we don't know
about the gps of any of the cities. Thus we filtered out cities with ratio < 0.3,
which means they are REALLY suspicious. With this threshold we are excluding the gps of
409 cities.

But this still leaves many problematic gps in the dataset, and our astar implementation
still gives results that are not optimal, if checked against the results of uniform search.
Therefore we decided to abandon the gps, and use the heuristics descibed in the beginning.

1.3 distance()
TODO

1.4 time()
TODO

2. RoadTrip():

This is the main class that solves the search problem.

3. Fringe():

In order to simplify our code, we make a Fringe class that is flexible enough to
handle all algorithms with minimum modification.

** The most important feature is that it can use priority queues with different 
comparators for 'time', 'distance' and 'segments'. **

4. MyHeap():

A more user friendly priority queue class, which takes in different comparators,
defined by lambda expressions. 

"""

# TODO check city-gps. Get the ratio of dist / crowFlyDis for all
# pairs in road-segments. If the ratio is too large, something
# must be wrong.

# TODO
# ./route.py Muncie,_Indiana Bellevue,_Washington astar distance
# now: 2195 miles
# ./route.py Muncie,_Indiana Bellevue,_Washington uniform distance
# 2158 miles

import sys, os, heapq, math
from math import radians

def main():
    if len(sys.argv) != 5:
        print('\nusage: \n\n./route.py [start-city] '
              '[end-city] [routing-algorithm] [cost-function]\n')
        sys.exit(1)

    startCityStr = sys.argv[1]
    endCityStr = sys.argv[2]
    routing = sys.argv[3]
    cost = sys.argv[4]

    mymap = MyMap()
    RT = RoadTrip(
        mymap=mymap,
        startCityStr=startCityStr,
        # startCityStr='Muncie,_Indiana',
        # startCityStr='Ritzville,_Washington',
        # startCityStr='Madison,_Wisconsin',
        # startCityStr='Bellevue,_Washington',
        # startCityStr='Bloomington,_Indiana',
        # endCityStr='Indianapolis,_Indiana',
        # endCityStr='Fort_Wayne,_Indiana',
        # endCityStr='Madison,_Wisconsin',
        # endCityStr='Bellevue,_Washington',  # 2158
        # endCityStr='Vantage,_Washington',
        # endCityStr='Boston,_Massachusetts',
        # endCityStr='Columbus,_Ohio',
        # endCityStr='Pittsburgh,_Pennsylvania',
        # endCityStr='Muncie,_Indiana',
        endCityStr=endCityStr,
        # routing='bfs',
        # routing='uniform',
        # routing='astar',
        routing=routing,
        # cost='distance',
        # cost='time',
        # cost='segments'
        cost=cost
    )

    # TODO piazza someone only visited 1545279 nodes,
    # but I visited 3654959 nodes; now I only visited 968689

    RT.solve()

def testHeap():
    """
    test if Fringe as a heap is removing the route_lst w/ smallest distance
    PASSED!
    """
    mymap = MyMap(test=True)
    fri = Fringe(mymap=mymap, routing='uniform', cost='distance')
    route1 = [ mymap.city2int['A'] ]
    route2 = [ mymap.city2int[i] for i in ['A','B','C'] ]
    route3 = [ mymap.city2int[i] for i in ['A','C','B'] ]
    route4 = [ mymap.city2int[i] for i in ['A','C','B','G'] ]
    route5 = [ mymap.city2int[i] for i in ['A','C','B','G','F'] ]
    route6 = [ mymap.city2int[i] for i in ['B','C'] ]
    fri.insert( route3 )
    fri.insert( route4 )
    fri.insert( route2 )
    fri.insert( route5 )
    fri.insert( route1 )
    fri.insert( route6 )
    print(fri.fringe._data)
    for i in range(len(fri)):
        x = fri.remove()
        print('\nremoved ', x)
        print(fri.fringe._data)

def testLatLon2Miles():
    """ test if we got correct crow-fly distance between two coords; PASSED! """
    mymap = MyMap()
    x=mymap.latLon2Miles((41.000833, -85.318056), (39.165325, -86.5263857))
    print(x)

def testGetCrowFlyDis():
    """ looks good """
    mymap = MyMap()
    mymap.fillcity2Coords()
    city1 = 'Indianapolis,_Indiana'
    city2 = 'Pittsburgh,_Pennsylvania'
    print(mymap.getCrowFlyDis(mymap.city2int[city1], mymap.city2int[city2]))

class MyMap(object):
    def __init__(self, test=False):
        self.MAP = {}       # cities represented by ints
        self.city2int = {}  # faster processing with ints
        self.int2city = {}
        self.build_MAP(test)
        # save distance of previous route_lst
        # key: tuple(route_lst), value: distance
        # self.distanceDict = {}
        self.city2Coords = {}  # {1 : (39.165325 -86.5263857)}
        self.fillcity2Coords()
        self.cities_wrong_gps_int = self.check()

    def fillcity2Coords(self):
        """ read in city-gps.txt and fill self.city2Coords """
        with open('city-gps.txt') as f:
            for line in f:
                line_l = line.strip().split(' ')
                if len(line_l) != 3: print(line_l)  # TODO
                cityStr = line_l[0]
                coords = (float(line_l[1]), float(line_l[2]))
                if cityStr in self.city2int:
                    if self.city2int[cityStr] in self.city2Coords:  # duplicate city entry
                        pass
                        # print('duplicate city:', line_l)  # ['Tatum,_New_Mexico', '33.2570566', '-103.317728']
                        # print(self.city2Coords[self.city2int[cityStr]])  # same coords!
                    else:
                        self.city2Coords[self.city2int[cityStr]] = coords
                else:
                    print('{:50} in city-gps, but not road-segments'.format(cityStr))

        # calculate crow-fly distance from 1 to 2,3,4... and store them in self.MAP
        # where i = 1, neighbor is the cityInt of 1's neighbors
        # THIS IS ACTUALLY SLOWER!!!
        # notInCityGps = []
        # for i in range(1, len(self.city2int) + 1):
        #     # check which cities are in road-segments, but not city-gps
        #     if i not in self.city2Coords:
        #         notInCityGps.append(i)
        #         print('{:50} in road-segments, but not city-gps'.format(self.int2city[i]))
        #     else:  # cities in both road-segments and city-gps
        #         self.crowFlyDis[i] = {}
        #         for j in range(i, len(self.city2int) + 1):
        #             # only set crowFlyDis if j also in city-gps
        #             if j in self.city2Coords:
        #                 # crowFlyDis = self.latLon2Miles(self.city2Coords[i],
        #                 #                                self.city2Coords[j])
        #                 crowFlyDis = 1
        #                 self.crowFlyDis[i][j] = crowFlyDis
        # print(len(notInCityGps), len(set(notInCityGps)))

    def getCrowFlyDis(self, city1int, city2int):
        # TODO if don't know the coords for city1 or city2, return 0 or something else?
        # print("{:20} {:20}".format(self.int2city[city1int], self.int2city[city2int]))
        if (city1int in self.cities_wrong_gps_int) or (city2int in self.cities_wrong_gps_int):
            # print('returing 0!')
            return 0
        if (city1int not in self.city2Coords) or (city2int not in self.city2Coords):
            return 0
        res = self.latLon2Miles( self.city2Coords[city1int], self.city2Coords[city2int] )
        # print(res)
        return res

    def latLon2Miles(self, latLonCity1, latLonCity2):
        '''
        compute crow-fly distance bet. 2 cities
        according to http://www.movable-type.co.uk/scripts/latlong.html
        latLonCity1 = (42.2411499 -83.6129939) tuple
        '''
        lat1 = latLonCity1[0]; lat2 = latLonCity2[0]
        lon1 = latLonCity1[1]; lon2 = latLonCity2[1]
        lat1Radius = radians(lat1)
        lat2Radius = radians(lat2)

        deltaLat = radians(lat2-lat1)
        deltaLon = radians(lon2-lon1)

        a = math.sin(deltaLat/2) * math.sin(deltaLat/2) + \
            math.cos(lat1Radius) * math.cos(lat2Radius) * \
            math.sin(deltaLon/2) * math.sin(deltaLon/2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        # Radius of earth = 3958.756  miles

        return 3958 * c * 0.9

    def time(self, route_lst):
        """ return time of a route_lst [3,2,25,6] """
        res = sum( [ self.time2cities( route_lst[i], route_lst[i+1] ) \
                      for i in range(len(route_lst)-1)] )
        return res

    def time2cities(self, city1int, city2int):
        return self.MAP[city1int][city2int]['time']

    def distance(self, route_lst):
        """ return distance of a route_lst [3,2,25,6] """
        # should we keep a dict of distances?? when will it be accessed?
        # seems to be slower than not keeping the dict
        # route_lst_tuple = tuple(route_lst)
        # if route_lst_tuple in self.distanceDict:
        #     return self.distanceDict[route_lst_tuple]
        res = sum( [ self.distance2cities( route_lst[i], route_lst[i+1] ) \
                      for i in range(len(route_lst)-1)] )
        # self.distanceDict[route_lst_tuple] = res
        return res

    def distance2cities(self, city1int, city2int):
        return self.MAP[city1int][city2int]['dist']

    def distNearestNeighbor(self, cityint):
        """ :return distance to nearest neighbor """
        return min( [ self.MAP[cityint][neib]['dist'] for neib \
                      in self.MAP[cityint].keys() ] )

    def timeShortestNeighbor(self, cityint):
        """ :return shortest time to any neighbor """
        return min([self.MAP[cityint][neib]['time'] for neib \
                    in self.MAP[cityint].keys()])

    def check(self):
        """ check which gps are wrong, return a list of city_ints that have wrong gps"""

        cities_wrong_gps_int = {}  # { cityname : numOfOccurrences }

        dist_dict = {}  # { (cityInt1, cityInt2) : {'road_dist': 10, 'crowFlyDist': 7.5, 'ratio': 3/4} }

        # for each two cities in road-segment, give the road mile, and crowFlyDis
        for city1Int in self.MAP.keys():
            for city2Int in self.MAP[city1Int].keys():
                road_dist = self.MAP[city1Int][city2Int]['dist']
                crowFlyDist = None
                try:
                    crowFlyDist = self.latLon2Miles(self.city2Coords[city1Int],
                                                     self.city2Coords[city2Int])
                except KeyError: pass

                # fill dict
                if crowFlyDist:
                    #####  important  #####
                    ratio = road_dist / crowFlyDist
                    #####  important  #####
                    x = min(city1Int, city2Int)
                    y = max(city1Int, city2Int)
                    dist_dict[(x, y)] = {'road_dist': road_dist,
                                         'crowFlyDist': crowFlyDist,
                                         'ratio': ratio}

        counter = 0
        for cityTuple, d in reversed(sorted(dist_dict.items(), key=lambda x: x[1]['ratio'])):
            ratio = dist_dict[cityTuple]['ratio']

            if (ratio <= 0.30) or (ratio > 20):  # TODO ratio threshold for wrong gps
                counter += 1
                cities_wrong_gps_int[cityTuple[0]] = cities_wrong_gps_int.get(cityTuple[0], 0) + 1
                cities_wrong_gps_int[cityTuple[1]] = cities_wrong_gps_int.get(cityTuple[1], 0) + 1

        # TODO find a threshold of ratio
        res = set([ city for city in cities_wrong_gps_int.keys() \
                     if cities_wrong_gps_int[city] > 0 ])
        print('check done! excluded cities:', len(res))
        return res

    def build_MAP(self, test):
        """
        read in road-segments.txt and build MAP, as a dict of dicts,
        {'Bloomington': {'Columbus' : {'dist':32, 'limit':45, 'highway': 'IN_46'},
                         'Martinsville': {'dist':... }, ... }, ... }
        """
        pairs = set()

        counter = 0
        fn = 'road-segments.txt'
        if test: fn = 'small_map'
        with open(fn) as f:
            for line in f:
                line_l = line.strip().split(' ')
                try: assert len(line_l) == 5
                except AssertionError:
                    print('bad format: {}'.format(line))
                    sys.exit(1)

                # Bloomington,_Indiana Columbus,_Indiana 32 45 IN_46
                try: start, end, dist, limit, highway = \
                    line_l[0], line_l[1], int(line_l[2]), int(line_l[3]), line_l[4]
                except ValueError:
                    # speed limit is unknown as in:
                    # Rexton,_New_Brunswick Shediac,_New_Brunswick 57  NB_11
                    # *** TODO 19 such cases ***
                    # print('bad format for distance or limit: {}'.format(line))
                    # start, end, dist, limit, highway = \
                    #     line_l[0], line_l[1], int(line_l[2]), 0, line_l[4]
                    # or ignore it?
                    continue

                if limit == 0: continue  # TODO 35 such cases
                time = dist / limit

                # add to map
                if start not in self.city2int:
                    counter += 1
                    self.city2int[start] = counter
                    self.int2city[counter] = start
                if end not in self.city2int:
                    counter += 1
                    self.city2int[end] = counter
                    self.int2city[counter] = end

                if self.city2int[start] not in self.MAP:
                    self.MAP[self.city2int[start]] = {}
                if self.city2int[end] not in self.MAP:
                    self.MAP[self.city2int[end]] = {}

                # city pair (start, end) already in MAP, two roads connecting start end!
                pair = tuple(sorted([self.city2int[start], self.city2int[end]]))
                if pair in pairs:
                    print('already in MAP:', self.int2city[pair[0]], self.int2city[pair[1]])
                    # 3 cases:
                    # Cutler_Ridge,_Florida Florida_City,_Florida 12 52 US_1
                    # Cutler_Ridge,_Florida Florida_City,_Florida 13 65 Florida's_Tpk
                    # Loxley,_Alabama Stapleton,_Alabama 4 52 AL_59
                    # Loxley,_Alabama Stapleton,_Alabama 4 52 AL_59
                    # North_Troy,_Vermont Troy,_Vermont 6 45 VT_101
                    # North_Troy,_Vermont Troy,_Vermont 6 45 VT_101

                    # choose the shorter path
                    oldDist = self.MAP[self.city2int[start]][self.city2int[end]]['dist']
                    if dist < oldDist:
                        self.MAP[self.city2int[start]][self.city2int[end]]['dist'] = dist
                        self.MAP[self.city2int[end]][self.city2int[start]]['dist'] = dist
                else: pairs.add(pair)

                self.MAP[self.city2int[start]][self.city2int[end]] = \
                    {'dist':dist,'limit':limit,'highway':highway,'time':time}
                self.MAP[self.city2int[end]][self.city2int[start]] = \
                    {'dist':dist,'limit':limit,'highway':highway,'time':time}

        print('MAP built successfully!')

        # sanity check
        # for city in ['Bloomington,_Indiana','Columbus,_Indiana']:
        #     print('\n\ncity: {} {}'.format(city, self.city2int[city]))
        #     for k, v in self.MAP[self.city2int[city]].items():
        #         print(k, v)

class RoadTrip(object):
    def __init__(self, mymap, startCityStr=None, endCityStr=None, routing=None, cost=None):
        self.mymap = mymap
        self.startCityStr = startCityStr
        self.endCityStr = endCityStr
        self.routing = routing
        self.cost = cost
        self.visited = set()
        self.numStatesChecked = 0

    def routeStr(self, routeInt):
        return [self.mymap.int2city[i] for i in routeInt]

    def successors(self, route):
        """
        if route is [1,3,2,7], and from 7 we can go to 10, 18, then
        we return a list of successors: [ [1,3,2,7,10], [1,3,2,7,18] ]
        """
        res = []
        lastCityInt = route[-1]
        for nextCityInt in self.mymap.MAP[ lastCityInt ].keys():
            res.append( route + [nextCityInt] )
        # print('\nsucc of {} are: '.format( self.routeStr(route) ))
        # for r in res: print( self.routeStr(r) )
        return res

    def is_goal(self, route, endCityInt):
        """ each route is a list [start, ... , end]; each step is int """
        return route[-1] == endCityInt

    def solve(self):
        """ this calls solve_main() for bfs, dfs, uniform, astar; calls solve_ids() for ids """
        if self.routing == 'ids': solution = self.solve_ids()
        else: solution = self.solve_main()

        print('\nnumStatesChecked: {}'.format(self.numStatesChecked))

        if self.cost == 'segments': optimal = 'yes'
        else:
            if self.routing in ['bfs', 'dfs', 'ids']: optimal = 'no'
            else: optimal = 'yes'

        outStr = '{} {} {} {}'
        print()

        if solution:
            mydistance = self.mymap.distance(solution)
            mytime = self.mymap.time(solution)
            mysolution = ' '.join(self.routeStr(solution))
            # print('\n\nsolution:', mysolution)
            print('\ndistance: {} miles'.format(mydistance))
            print('time: {} hrs'.format(mytime))
            print('segments: {} turns\n'.format(len(solution)))
            print(outStr.format(optimal, mydistance, mytime, mysolution))

        else:
            print('\n\nno solution')

    def solve_ids(self):
        print('\nfinding route from *{}* to *{}*, using *{}*, where cost = *{}* '
              '...'.format(self.startCityStr, self.endCityStr, self.routing, self.cost))
        solution = None
        depth = 1
        while not solution:
            solution = self.solve_ids_helper(depth)
            depth += 1
        return solution

    def solve_ids_helper(self, depth):
        print('ids w/ depth {}'.format(depth))

        endCityInt = self.mymap.city2int[self.endCityStr]
        init_route = [self.mymap.city2int[self.startCityStr]]
        fri = Fringe(mymap=self.mymap, routing=self.routing, cost=self.cost, endCityInt=endCityInt)
        fri.insert(init_route)

        while len(fri) > 0:
            removed = fri.remove()
            # print(removed)
            for succ in self.successors(removed):  # succ = [2,3,4]
                if self.is_goal(succ, endCityInt): return succ
                self.numStatesChecked += 1
                if len(succ) <= depth:
                    fri.insert(succ)
        return False

    def solve_main(self):
        """
        accepting different 'routing' and 'cost'
        """
        print('\nfinding route from *{}* to *{}*, using *{}*, where cost = *{}* '
              '...'.format(self.startCityStr, self.endCityStr, self.routing, self.cost))

        # in our implementation  BFS and UCS  should use
        # exactly the same code? b/c the only difference is
        # we are popping the state with least distance??
        # see slide 47 from : http://www.ccs.neu.edu/home/rplatt/cs5335_2016/slides/bfs_ucs.pdf

        endCityInt = self.mymap.city2int[ self.endCityStr ]

        init_route = [ self.mymap.city2int[self.startCityStr] ]
        fri = Fringe(mymap=self.mymap, routing=self.routing, cost=self.cost, endCityInt=endCityInt)
        fri.insert( init_route )

        while len(fri) > 0:

            removed = fri.remove()
            self.numStatesChecked += 1
            self.visited.add(removed[-1])  # TODO where to add to visited???
            # print(removed)

            if self.routing in {'uniform', 'astar'}:
                # check is_goal here
                if self.is_goal(removed, endCityInt): return removed

            for succ in self.successors( removed ):  # succ = [2,3,4]
                if self.routing in {'bfs', 'dfs'}:  # check is_goal here
                    if self.is_goal(succ, endCityInt): return succ

                if self.routing in {'uniform', 'astar'}:
                    if succ[-1] not in self.visited:
                        fri.insert(succ)
                        # self.visited.add(succ[-1])  # TODO where to add to visited???

                elif self.routing in {'bfs', 'dfs'}:
                    fri.insert(succ)
        return False

class Fringe(object):
    def __init__(self, mymap, routing=None, cost=None, endCityInt=None):
        self.routing = routing
        self.cost = cost
        self.mymap = mymap

        if routing in {'bfs', 'dfs', 'ids'}:
            self.fringe = []
        elif routing == 'uniform':
            if cost == 'distance':
                self.fringe = MyHeap(key=lambda \
                        route_lst : self.mymap.distance(route_lst))
            elif cost == 'time':
                self.fringe = MyHeap(key=lambda \
                        route_lst : self.mymap.time(route_lst))
            elif cost == 'segments':
                self.fringe = MyHeap(key=lambda route_lst : len(route_lst))
        elif routing == 'astar':
            if cost == 'distance':
                # h = crow-fly distance between x to endcity (use gps to calculate)
                # which is very inaccurate
                # print('*** crow-dist as heuristics ***')
                # self.fringe = MyHeap(key=lambda \
                #         route_lst : self.mymap.distance(route_lst) +
                #                     self.mymap.getCrowFlyDis(route_lst[-1], endCityInt))

                # or h = distance to nearest neighbor
                # slower but definitely admissible
                print('*** nearest neighbor as heuristics ***')
                self.fringe = MyHeap(key=lambda \
                        route_lst : self.mymap.distance(route_lst) +
                                    self.mymap.distNearestNeighbor(route_lst[-1]))

            elif cost == 'time':  # TODO
                # h = shortest time to a neighbor
                self.fringe = MyHeap(key=lambda \
                        route_lst : self.mymap.time(route_lst) +
                                    self.mymap.timeShortestNeighbor(route_lst[-1]))

            elif cost == 'segments':  # TODO
                self.fringe = MyHeap(key=lambda route_lst : len(route_lst) + 1)

    def __len__(self):
        if self.routing in {'bfs', 'dfs', 'ids'}:
            return len(self.fringe)
        elif self.routing in {'uniform', 'astar'}:
            return len(self.fringe._data)

    def remove(self):
        if self.routing == 'bfs':  # pop from beginning
            res = self.fringe.pop(0)
        elif self.routing in {'dfs', 'ids'}:  # pop from end
            res = self.fringe.pop()
        elif self.routing in {'uniform', 'astar'}:
            res = self.fringe.pop()
        return res

    def insert(self, route_lst):
        if self.routing in {'bfs', 'dfs', 'ids'}:
            self.fringe.append(route_lst)
        elif self.routing in {'uniform', 'astar'}:
            self.fringe.push(route_lst)


class MyHeap(object):
    """
    a more user-friendly heap
    taken from: https://stackoverflow.com/questions/8875706/heapq-with-custom-compare-predicate

    for cost=distance, the key we use is:
    key=lambda route_lst : self.mymap.distance(route_lst)
    """
    def __init__(self, initial=None, key=lambda x:x):
        self.key = key
        if initial:
            self._data = [(key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self._data)[1]

if __name__ == '__main__':
    main()
