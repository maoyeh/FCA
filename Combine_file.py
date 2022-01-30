
import os
import pandas as pd
import numpy as np
import pylab as plt

path = input('Input data path:   ')
data = []
for f in os.listdir(path):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(path,f), header=0, index_col=0)
        data.append(df.iloc[0])

data= np.array(data)

fig = plt.figure()
axes = fig.subplots()

time = np.linspace(0, (5 + 25) * 60 , len(data[0]), endpoint=True)
data_mean = np.mean(data, axis=0)
error = np.std(data,axis=0) / np.sqrt(data.shape[0])

plt.plot(time,data_mean, 'k-')
plt.fill_between(time,data_mean - error, data_mean + error)
axes.set_xlabel('Time (s)')
axes.set_ylabel('Fluorescence')
plt.savefig(os.path.join(path,'average_error.png'))
plt.savefig(os.path.join(path,'average_error.eps'))
plt.show()

