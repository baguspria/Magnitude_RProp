import numpy as np

def grid_num(lats, longs, x_grids, y_grids):
    #calc length and width of one grid
    length = (np.max(lats)-np.min(lats)) / y_grids
    width = (np.max(longs)-np.min(longs)) / x_grids

    #create a list of lats&longs with calculated steps
    lat_grids = [i for i in np.arange(np.min(lats), np.max(lats)+1, length)]
    long_grids = [i for i in np.arange(np.min(longs), np.max(longs)+1, width)]

    #create zip of min-max for each longs&lats
    lat_grids = np.array(list(zip(lat_grids, lat_grids[1:])))
    long_grids = np.array(list(zip(long_grids, long_grids[1:])))
    lat_grids[1:,0] = lat_grids[1:,0] + 0.001
    long_grids[1:,0] = long_grids[1:,0] + 0.001
    
    #create zip of longs-lats
    return [(i, j) for i in lat_grids for j in long_grids]

