import dowhy
from dowhy import CausalModel

import numpy as np
import pandas as pd
import graphviz
import networkx as nx

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)
data_mpg = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original',
                       delim_whitespace=True, header=None,
                       names=['mpg', 'cylinders', 'displacement',
                              'horsepower', 'weight', 'acceleration',
                              'model year', 'origin', 'car name'])
data_mpg.dropna(inplace=True)
data_mpg.drop(['model year', 'origin', 'car name'], axis=1, inplace=True)
print(data_mpg.shape)
data_mpg.head()

labels = [f'{col}' for i, col in enumerate(data_mpg.columns)]
data = data_mpg.to_numpy()
from causallearn.search.FCMBased import lingam

model = lingam.ICALiNGAM()
model.fit(data)

from causallearn.search.FCMBased.lingam.utils import make_dot

make_dot(model.adjacency_matrix_, labels=labels)