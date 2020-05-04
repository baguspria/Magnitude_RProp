#imports
import numpy as np
import pandas as pd
import math

#override round (round up if val >= 5)
def round(x, dec=0):
    mult = 10**dec
    return (math.floor(x * mult + 0.5)) / mult

#returns df of selected col indexes from csv
def fetch_data(dir, filenames, col_idx):
    return pd.concat([pd.read_csv(dir+file, usecols=col_idx) for file in filenames])

#returns grid numbers w/ boundaries
def grid_num(lats, longs, x_grids, y_grids, max, min):
    #calc length and width of one grid
    length = (max[0]-min[0]) / y_grids
    width = (max[1]-min[1]) / x_grids

    #create a list of lats & longs with calculated steps
    lat_grids = np.arange(min[0], max[0]+1, length)
    long_grids = np.arange(min[1], max[1]+1, width)
    
    lat_add = np.arange(0.001, 0.001 * (len(lat_grids)-2), 0.001)
    long_add = np.arange(0.001, 0.001 * (len(long_grids)-2), 0.001)
    
    lat_grids[2:-1]+=lat_add
    long_grids[2:-1]+=long_add
    long_grids = [round(i,3) for i in long_grids]

    #create zip of min-max for each longs&lats
    lat_grids = np.array(list(zip(lat_grids, lat_grids[1:])))
    long_grids = np.array(list(zip(long_grids, long_grids[1:])))
    
    lat_grids = np.flip(lat_grids, 0)
    
    lat_grids[1:,1] = lat_grids[1:,1] + 0.001
    long_grids[1:,0] = long_grids[1:,0] + 0.001
    
    
    #create zip of longs-lats
    return pd.DataFrame([np.append(i, j) for i in lat_grids for j in long_grids], columns=['minlat','maxlat','minlong','maxlong'])

#returns normalized data
def minmax_norm(arr):
    min = np.min(arr)
    diff = np.max(arr)-min
    return [(x-min)/diff for x in arr]

#---------------------M A I N-----------------------#

#fetching raw data
dir = 'USGS datasets/'
filenames = ['2000-2005.csv', '2006-2013.csv', '2014-2019.csv']
col_idx = [0,1,2,4]
raw = fetch_data(dir, filenames, col_idx)

#grid numbering
lat = np.array(raw['latitude'])
long = np.array(raw['longitude'])
grids = grid_num(lat, long, 8, 2, [5.907, 140.976], [-10.909, 95.206])
print(grids)

#min-max normalization
norm_mag = minmax_norm(raw['mag'])

#create un-grouped dataset
# extracted = pd.DataFrame([raw['time'], grids, ])




