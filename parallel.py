# this graph is unreadable

import pandas
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

from load import data, em_data

gt_data = []

for i in range(len(data)):
    gt_data.append(data[i][2])     # extracts the ground truth column
    
    if i != 0:
        em_data[i].insert(0, int(gt_data[i]))
    else:
        em_data[i].insert(0, gt_data[i]) # insterts heading as a string

em_df = pandas.DataFrame(em_data[1:], columns = em_data[0])

pandas.plotting.parallel_coordinates(em_df, class_column="gt")
plt.show()