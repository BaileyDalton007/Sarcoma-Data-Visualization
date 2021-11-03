# this graph is unreadable

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

from load import getEmbedData

gt_data = []

em_data, data = getEmbedData()

for i in range(len(data)):
    gt_data.append(data[i][2])     # extracts the ground truth column
    
    if i != 0:
        em_data[i].insert(0, int(gt_data[i]))
    else:
        em_data[i].insert(0, gt_data[i]) # insterts heading as a string

em_df = pd.DataFrame(em_data[1:], columns = em_data[0])

sub_df = em_df.iloc[0:100,] # copies embed data to be cut down and graphed

pd.plotting.parallel_coordinates(sub_df, class_column="gt")
plt.show()