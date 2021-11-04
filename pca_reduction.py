import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from load import getData

# amount of random points plotted
images, gt, pred_class, pred_prob, probs, features, columns, file_name = getData(0.9)

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(features.shape[0])

pca = PCA(n_components=3)
pca_result = pca.fit_transform(features[columns].values)
features['pca-one'] = pca_result[:,0]
features['pca-two'] = pca_result[:,1] 
features['pca-three'] = pca_result[:,2]

# 2-D Graph
#plt.figure(figsize=(16,10))
#sns.scatterplot(
#    x="pca-one", y="pca-two",
#    hue=pred_class,
#    palette=sns.color_palette("hls", 3),
#    data=features.loc[rndperm,:],
#    legend="full",
#    alpha=0.3
#).set(title=file_name)

# 3-D Graph
color_map = []
for i in range(len(pred_class)):
    if int(pred_class[i]) == 0:
        color_map.append((1, 0, 0, 1))
    if int(pred_class[i]) == 1:
        color_map.append((0, 1, 0, 0.5))
    if int(pred_class[i]) == 2:
        color_map.append((0, 0, 1, 1))

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=features.loc[rndperm,:]["pca-one"], 
    ys=features.loc[rndperm,:]["pca-two"], 
    zs=features.loc[rndperm,:]["pca-three"], 
    c=color_map, 
    cmap='tab3'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')

ax.set_title(file_name)

plt.show()