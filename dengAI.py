# %matplotlib inline

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')


# Load the provided data
train_features = pd.read_csv('dengue_features_train.csv', index_col=[0,1,2])
train_labels = pd.read_csv('dengue_labels_train.csv', index_col=[0,1,2])

# Seperate data for San Juan
sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

# Seprate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']

print('San Juan')
print('features: ', sj_train_features.shape)
print('labels  : ', sj_train_labels.shape)

print('\nIquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)

print(sj_train_features.head())

# Remove `week_start_date`string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)


print(pd.isnull(sj_train_features).any())

# to remove gaps where there are NaNs by ploting in data
# we'll take a simple approach and just fill those values 
# with the most recent value that we saw up to that point.
# If you want see this gaps comment the two line below 
# and plot the graph again
sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)

(sj_train_features
    .ndvi_ne
    .plot
    .line(lw=0.8))
    
plt.title('Vegetation Index over Time')
plt.xlabel('Time')


plt.show()
