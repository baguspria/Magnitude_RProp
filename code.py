#   TO DO:
# - Create class
# - Make sure no hardcode variables (make it accessible anywhere, and not fail on any type of data)

#imports
import numpy as np
import pandas as pd
import math, datetime

#override round (round up if val >= 5)
def round(x, dec=0):
    mult = 10**dec
    return (math.floor(x * mult + 0.5)) / mult

#returns df of selected col indexes from csv
def fetch_data(dir, filenames, col_idx):
    return pd.concat([pd.read_csv(dir+file, usecols=col_idx) for file in filenames],ignore_index=True)

#returns grid numbers w/ boundaries
def grid_num(lats, longs, x_grids, y_grids, maxs, mins):
    #calc length and width of one grid
    length = (maxs[0]-mins[0]) / y_grids
    width = (maxs[1]-mins[1]) / x_grids

    #create a list of lats & longs with calculated steps
    lat_grids = np.arange(mins[0], maxs[0]+1, length)
    long_grids = np.arange(mins[1], maxs[1]+1, width)
    
    lat_add = np.arange(0.001, 0.001 * (len(lat_grids)-2), 0.001)
    long_add = np.arange(0.001, 0.001 * (len(long_grids)-2), 0.001)
    
    lat_grids[2:-1]+=lat_add
    long_grids[2:-1]+=long_add
    long_grids = [round(i,3) for i in long_grids]

    #create zip of min-max for each longs&lats
    lat_grids = np.array(list(zip(lat_grids, lat_grids[1:])))
    long_grids = np.array(list(zip(long_grids, long_grids[1:])))
    
    #read lat top-bottom
    lat_grids = np.flip(lat_grids, 0)
    
    lat_grids[1:,1] = lat_grids[1:,1] + 0.001
    long_grids[1:,0] = long_grids[1:,0] + 0.001

    grids = pd.DataFrame([np.append(i, j) for i in lat_grids for j in long_grids], columns=['min_lat','max_lat','min_long','max_long'], index=[i for i in range(1, 17)])
    data = pd.DataFrame([lats, longs]).transpose().rename(columns={0:'latitude', 1:'longitude'})
    
    #grid numbering
    grid_num = []
    lat_mins = np.flip(lat_grids[:,0])
    long_mins = long_grids[:,0]

    for x in data.itertuples():
        mask = (grids['min_lat']==lat_mins[np.searchsorted(lat_mins, x[1], side='right')-1]) & (grids['min_long']==long_mins[np.searchsorted(long_mins, x[2], side='right')-1])
        grid_num.append(grids.index[mask][0])

    #concat with grid_num
    data = pd.concat([data, pd.DataFrame(grid_num, columns=['grid_num'])], axis=1)
    
    return data, grids

#returns normalized data
def minmax_norm(arr):
    min = np.min(arr)
    diff = np.max(arr)-min
    return [(x-min)/diff for x in arr]

#returns date-pair
def date_pair(time):
    ids=[0]
    while True:
        x = np.searchsorted(time, time[ids[-1]]+datetime.timedelta(days=6), side='right')
        ids.append(x)
        if x >= len(time): break
        

    start_date = time.reindex(ids).reset_index(drop=True)
    end_date = time.reindex(np.array(ids[1:])-1).reset_index(drop=True)

    date_list = pd.concat([start_date, end_date], ignore_index=True, axis=1).dropna()
    date_list.columns = ['start_date', 'end_date']
    return date_list, ids

def preprocess(dir, filename, col_idx):
    #fetching raw data
    dir = 'USGS datasets/'
    filenames = ['2000-2005.csv', '2006-2013.csv', '2014-2019.csv']
    col_idx = [0,1,2,4]
    raw = fetch_data(dir, filenames, col_idx)

    #get formatted dates only
    raw['time'] = raw['time'].apply(lambda x: datetime.datetime.strptime(x[:10], "%Y-%m-%d"))

    #grid numbering
    lat = np.array(raw['latitude'])
    long = np.array(raw['longitude'])
    nummed_grids, boundaries = grid_num(lat, long, 8, 2, [5.907, 140.976], [-10.909, 95.206])

    #min-max normalization
    norm_mag = pd.DataFrame(minmax_norm(raw['mag']), columns=['norm_mag'])

    #helper dataset
    raw = pd.concat([raw['time'], nummed_grids['grid_num'], norm_mag], axis=1)

    #create final dataset
    dates, helper_ids = date_pair(raw['time'])
    grids = boundaries.index
    arr = []

    for date in dates.itertuples():
        masked = raw[(raw['time']>=date[1]) & (raw['time']<=date[2])]
        arr.append([masked['norm_mag'][masked['grid_num']==x].mean() for x in grids])

    grids = pd.Series(np.tile(grids, len(dates)), name='grid').astype(int)
    dates = dates.loc[dates.index.repeat(len(boundaries.index))].reset_index(drop=True)
    avg_mag = pd.Series(np.array(arr).flatten(), name='avg_mag')
    target = pd.Series(avg_mag.iloc[len(boundaries.index):], name='target').reset_index(drop=True)

    data = pd.concat([grids, dates, avg_mag, target], axis=1).fillna(0)
    data = data[:-len(boundaries.index)]

    # print(data)
