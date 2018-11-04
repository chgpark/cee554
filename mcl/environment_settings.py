
#
#


'''Pohang Settings'''
# UWB_LOC = [[ 23.8, 0.55, 0.01], [ 19.2, 7.8, 0.01],
#            [ 16.5, 0.54, 0.01], [ 12.2, 2.7, 0.01],
#            [ 12.2,  0.1, 0.01], [  4.2, 0.12, 0.01],
#            [ 5.3, 8.2, 0.01], [0.56, 0.53, 0.01],
#            [ 0.4, 11.3, 0.01], [ 5.2, 13.4, 0.01],
#            [ 6.0, 16.0, 0.01], [ 12.1, 16.1, 0.01],
#            [ 12.1, 13.5, 0.01], [ 19.2, 16.1, 0.01],
#            [ 19.2, 13.1, 0.01], [23.8, 11.3, 0.01]]


'''Node for Pohang TEST'''
# INSIDE_NODES = ((5.3, 13.4), (5.3, 2.8), (19.0, 2.8), (19.0, 13.4))
# INITIAL_EDGES = [[   0, None,  None, None, None,   -1],
#                  [None,     0, 10.6,   -1, 13.7, None],
#                  [None,  10.6,    0, 13.7,   -1, None],
#                  [None,    -1, 13.7,    0, 10.6, None],
#                  [None,  13.7,   -1, 10.6,    0, None],
#                  [  -1, None,  None, None, None,    0]]

# MAP_X = 24.0
# MAP_Y = 16.0
# MAP_Z = 3.0

'''180806 experiment setting'''
# UWB_LOC = [[0.0, 0.0, 0.02 ], [7.2 , 0.0, 0.75],
#           [7.2, 2.70, 0.03 ], [0.0,  2.70, 1.1]] # temporary location input
#
# MAP_X = 7.2
# MAP_Y = 2.7
# MAP_Z = 3.0
# # BIAS = 0.0
#
# '''Variables of multipath_checker'''


'''180903 multipath 12 anchor'''
INSIDE_NODES = (
    (    0, 1.61),
    (16.87, 1.61),
    (16.87,  8.0))

UWB_LOC = [[-0.877,  0.401, 0.122], [-1.313, -1.429, 1.519],
           [-0.173, -0.519, 0.185], [ 0.687, -0.638, 0.188],
           [ 1.340, -0.565, 1.550], [ 1.387, 1.205, 0.133],
           [ 0.638,  1.436, 1.437], [-0.613, 0.854, 1.484]]

INITIAL_EDGES = [[   0,  None, None, None,   -1],
                 [None,     0, 16.87,   -1, None],
                 [None, 16.87,     0, 6.39, None],
                 [None,    -1,  6.39,    0, None],
                 [  -1,  None, None, None,     0]]

MAP_X = 6
MAP_Y = 6
MAP_Z = 3

'''180808'''
# UWB_LOC = [[0.0, 0.0, 0.02], [7.2, 0, 0.02],
#            [7.2, 2.7, 0.02], [0.0, 2.7, 0.02]]
# MAP_X = 7.2
# MAP_Y = 2.7
# MAP_Z = 3
'''Variables of MCL'''
SAMPLING_NUM = 1000
UNCERTAINTY = 0.094
SENSOR_NOISE = 0.01

'''Variables for LiDAR'''
MEASUREMENT_LiDAR_WEIGHT = 10
DIFFERENCE_TAG_LiDAR = 0.43
INITIAL_HEIGHT = 0.5





