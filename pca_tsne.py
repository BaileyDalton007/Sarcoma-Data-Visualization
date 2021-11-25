import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from load import getData

def reduction(features, columns):
    # amount of random points plotted

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(features[columns].values)

    tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=300, random_state=0)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    features['tsne-pca50-one'] = tsne_pca_results[:,0]
    features['tsne-pca50-two'] = tsne_pca_results[:,1]

    return features

def showGraph():
    images, gt, pred_class, pred_prob, probs, features, columns, file_name = getData(0.99)
    features = reduction(features, columns)

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue=pred_class,
        palette=sns.color_palette("hls", 3),
        data=features,
        legend="full",
        alpha=0.3
    ).set(title=file_name)
    plt.show()
