import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from load import getData

# amount of random points plotted
images, gt, pred_class, pred_prob, probs, features, columns, file_name = getData(0.9)

tsne = TSNE(n_components=2, verbose=1, perplexity=60, n_iter=300, random_state=0) # random state constant for reproducability
tsne_results = tsne.fit_transform(features[columns].values)

features['tsne-2d-one'] = tsne_results[:,0]
features['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue=pred_class,
    palette=sns.color_palette("hls", 3),
    data=features,
    legend="full",
    alpha=0.3
).set(title=file_name)

plt.show()