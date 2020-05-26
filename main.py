#%%
from preprocessing import Preprocessing
from rprop import RProp
import numpy as np
import time
import matplotlib.pyplot as plt
#%%
dir = 'USGS datasets/'
filenames = ['2000-2005.csv', '2006-2013.csv', '2014-2019.csv']
col_idx = [0, 1, 2, 4]

pp = Preprocessing()
data = pp.get_data(dir, filenames, col_idx)

train = data[data['start_date'] < "2019-01-01"]
test = data[data['start_date'] >= "2019-01-01"]

x_train = np.reshape(train.avg_mag.values, (len(train) // len(np.unique(train.grid)), len(np.unique(train.grid))))
y_train = np.reshape(train.target.values, (len(train) // len(np.unique(train.grid)), len(np.unique(train.grid))))
x_test = np.reshape(test.avg_mag.values, (len(test) // len(np.unique(test.grid)), len(np.unique(test.grid))))
y_test = np.reshape(test.target.values, (len(test) // len(np.unique(test.grid)), len(np.unique(test.grid))))

#%%
acc = np.empty((10, 46))
t = np.empty((10,46))
bottom = 50
up = 501
step = 10

for i in range (10):
    print('i = {:d}\n{:s}'.format(i, '-'*50))
    for j in range(bottom, up, step):
        print('j = {:d}'.format(j), end=' // ')
        nn = RProp([16,5,16])        
        start = time.time()
        nn.train(x_train, y_train, max_epoch=j)
        train_time = time.time()
        pred = nn.test(x_test)
        test_time = time.time()
        acc[i][(j-bottom)//step] = nn.accuracy(pred, y_test)
        t[i][(j-bottom)//step] = train_time-start

#%%
import pandas as pd
df = pd.DataFrame(acc, columns=[j for j in range(bottom, up, step)])
df.to_csv('Tests 2/acc1')


# plt.plot([i for i in range (1,11)], acc[:,2])
# plt.xlabel('Jumlah Epoch')
# plt.ylabel('Akurasi')
# plt.xticks([i for i in range (1,11)])
# # plt.yticks(np.arange(min(arr[:,0]), max(arr[:,0])+10**-5, 10**-6))
# plt.show()


# fig, axs = plt.subplots(3)
# axs[0].plot([i for i in range (100, 1001, 10)], arr[:,0])
# axs[0].set_title('Accuracy')
# axs[1].plot([i for i in range (100, 1001, 10)], arr[:,1])
# axs[0].set_title('Training time')
# axs[2].plot([i for i in range (100, 1001, 10)], arr[:,2])
# axs[0].set_title('Testing time')