from preprocessing import Preprocessing

dir = 'USGS datasets/'
filenames = ['2000-2005.csv', '2000-2006.csv', '2014-2019.csv']
col_idx = [0, 1, 2, 4]

pp = Preprocessing()

data = pp.get_data(dir, filenames, col_idx)

