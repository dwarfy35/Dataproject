#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

file_input = input("Enter the directory of the file: ")
cluster_input = int(input("How many clusters do you want?"))
data = pd.read_csv(file_input, sep='\t', index_col=0)
data = data.loc[:, data.columns != 'cancer']
name_of_file = input("What would you like to name the file?")


def pca_fit(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    data_pca = pca.transform(data)

    return data_pca


def kmeans_cluster(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.savefig(name_of_file + '.png')


pca_fit(data)

kmeans_cluster(pca_fit(data), cluster_input)
