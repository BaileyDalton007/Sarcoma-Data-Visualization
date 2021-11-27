from  load import getData
from pca_reduction import reduction
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot

images, gt, pred_class, pred_prob, probs, features, columns, file_name = getData(0.99, 1000)
features = reduction(features, columns)
print(len(features))


X = np.array(list(zip(features['pca-one'], features['pca-two'])))

model = AgglomerativeClustering(n_clusters=3)

yhat = model.fit_predict(X)
clusters = np.unique(yhat)

for cluster in clusters:
	row_ix = np.where(yhat == cluster)
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
pyplot.show()


