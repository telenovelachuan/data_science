import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width

# K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8)
y_pred = kmeans.fit_predict(X)
print "centers:{}".format(kmeans.cluster_centers_)
print "y_pred:{}".format(y_pred)


def plot_kmeans_clustering(model, X):
    def generate_random_hex():
        import random
        HEX_SYMBOLS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f']
        result = "#"
        for i in range(6):
            result += HEX_SYMBOLS[random.randint(0, 15)]
        return result
    def generate_random_marker():
        MARKER_SYMBOLS = ['.', '^', 's', 'P', '*', 'x']
        import random
        return MARKER_SYMBOLS[random.randint(0, len(MARKER_SYMBOLS) - 1)]

    y_pred = model.fit_predict(X)
    for y_class in set(y_pred):
        plt.scatter(
            X[y_pred == y_class, 0], X[y_pred == y_class, 1],
            s=50, c=generate_random_hex(),
            marker=generate_random_marker(), edgecolor='black',
            label='cluster 1'
        )
    plt.show()


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    import numpy as np

    def plot_data(x):
        plt.plot(x[:, 0], x[:, 1], 'k.', markersize=2)

    def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
        if weights is not None:
            centroids = centroids[weights > weights.max() / 10]
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='o', s=30, linewidths=8,
                    color=circle_color, zorder=10, alpha=0.9)
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=50, linewidths=50,
                    color=cross_color, zorder=11, alpha=1)

    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.show()

# plot_kmeans_clustering(kmeans, X)
# plot_decision_boundaries(kmeans, X)


# judging number of clusters
silhouette_scores = []
from sklearn.metrics import silhouette_score
for i in range(20)[2: ]:
    km = KMeans(n_clusters=i)
    km.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, km.labels_))

# plt.plot(silhouette_scores)
# plt.show()
print "silhouette_scores:{}".format(silhouette_scores)


def plot_silhouette_histogram(X):
    from sklearn.metrics import silhouette_samples
    from matplotlib.ticker import FixedLocator, FixedFormatter
    import matplotlib as mpl
    import numpy as np

    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                    for k in range(1, 10)]
    inertias = [model.inertia_ for model in kmeans_per_k]
    plt.figure(figsize=(14, 7))

    for k in (3, 4, 5, 6):
        plt.subplot(2, 2, k - 2)

        y_pred = kmeans_per_k[k - 1].labels_
        silhouette_coefficients = silhouette_samples(X, y_pred)

        padding = len(X) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = mpl.cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in (3, 5):
            plt.ylabel("Cluster")

        if k in (5, 6):
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.tick_params(labelbottom=False)

        plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)

    plt.show()


# plot_silhouette_histogram(X)


# DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.05)
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X)
# print "dbscan.labels_:{}".format(dbscan.labels_)
plot_kmeans_clustering(dbscan, X)


