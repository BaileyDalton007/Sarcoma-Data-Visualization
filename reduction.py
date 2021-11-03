import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from load import getEmbedData

# amount of random points plotted
max_size = 30000
em_data, data = getEmbedData(0.99)
gt_data = []

for i in range(len(data)):
    gt_data.append(data[i][2])     # extracts the ground truth column

em_data = em_data[:max_size]
em_df = pd.DataFrame(em_data[1:], columns = em_data[0])

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(em_df.shape[0])

pca = PCA(n_components=3)
pca_result = pca.fit_transform(em_df[em_data[0]].values)
em_df['pca-one'] = pca_result[:,0]
em_df['pca-two'] = pca_result[:,1] 
em_df['pca-three'] = pca_result[:,2]

# 2-D Graph
#plt.figure(figsize=(16,10))
#sns.scatterplot(
#    x="pca-one", y="pca-two",
#    hue=gt_data[1:],
#    palette=sns.color_palette("hls", 3),
#    data=em_df.loc[rndperm,:],
#    legend="full",
#    alpha=0.3
#)

# 3-D Graph
color_map = []
for i in range(len(gt_data[:max_size]) - 1):
    if int(gt_data[i + 1]) == 0:
        color_map.append((1, 0, 0, 1))
    if int(gt_data[i + 1]) == 1:
        color_map.append((0, 1, 0, 0.5))
    if int(gt_data[i + 1]) == 2:
        color_map.append((0, 0, 1, 1))

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=em_df.loc[rndperm,:]["pca-one"], 
    ys=em_df.loc[rndperm,:]["pca-two"], 
    zs=em_df.loc[rndperm,:]["pca-three"], 
    c=color_map, 
    cmap='tab3'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()